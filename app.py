"""
Lot Scout — used-car listing analyzer (HF Build Small Hackathon · Backyard AI track)

A buyer uploads a listing screenshot OR pastes text; Lot Scout runs a visible agent
pipeline entirely on-device (MiniCPM-V 4.6, ~1.3B params, Apache-2.0) and returns:
facts · red flags · a fair-price ESTIMATE · 5 seller questions · a walk-away price.

Pipeline (each step logs its I/O to a trace):
  extract (vision model call)  ->  validate/normalize (rules)  ->  red-flag check (rules)
  ->  price sanity (depreciation heuristic)  ->  advise (rules)  ->  assemble
Only the extract step calls the model: a ~1.3B model reads images well but recalls
market prices and free-form judgment poorly, so pricing/advice are deterministic.

No cloud LLM APIs. Inference path verified against the running official demo Space
openbmb/MiniCPM-V-4.6-Demo: AutoProcessor + MiniCPMV4_6ForConditionalGeneration +
processor.apply_chat_template(...) + model.generate(...)  (there is no model.chat()).
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import AutoProcessor, MiniCPMV4_6ForConditionalGeneration

# --------------------------------------------------------------------------- #
# Model loading (module level so ZeroGPU keeps it warm across calls)
# --------------------------------------------------------------------------- #
MODEL_ID = "openbmb/MiniCPM-V-4.6"
GPU_DURATION = 60  # ZeroGPU lease seconds; full pipeline target is <10s

print(f"[lot-scout] loading processor: {MODEL_ID}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"[lot-scout] loading model: {MODEL_ID}", flush=True)
model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
    device_map="cuda",
).eval()


# --------------------------------------------------------------------------- #
# Core inference helper (verified signature — see module docstring)
# --------------------------------------------------------------------------- #
def _run_model(content: list[dict], max_new_tokens: int = 512) -> str:
    """Single non-streaming greedy pass. enable_thinking=False for speed/determinism."""
    messages = [{"role": "user", "content": content}]
    has_image = any(item.get("type") == "image" for item in content)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs={
            "downsample_mode": "16x",
            "max_slice_nums": 9 if has_image else 1,
            "use_image_id": has_image,
        },
    ).to(model.device)

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            inputs[key] = value.to(dtype=torch.bfloat16)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        downsample_mode="16x",
    )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def _parse_json(raw: str) -> dict | None:
    """Extract the first *balanced* JSON object, tolerating trailing junk / stray braces."""
    start = raw.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------- #
# Pipeline trace — exportable later for the Open Trace badge
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, inp: Any, out: Any, ms: float) -> None:
        self.steps.append({"step": name, "input": inp, "output": out, "ms": round(ms, 1)})

    @property
    def total_ms(self) -> float:
        return sum(s["ms"] for s in self.steps)


# --------------------------------------------------------------------------- #
# STEP 1 — EXTRACT (vision model call)
# --------------------------------------------------------------------------- #
EXTRACT_PROMPT = """You are reading a used-car listing (screenshot and/or text). Extract facts as STRICT JSON, keys exactly:
{"year": int|null, "make": str|null, "model": str|null, "trim": str|null, "mileage": int|null, "price": int|null, "title_status": str|null, "location": str|null, "seller_notes": str|null}
Rules:
- "year" = the 4-digit model year (e.g. 2013). Pull it out of the title even if combined with make and model.
- "make" = brand only (e.g. Honda). "model" = model name only (e.g. Accord). "trim" = trim/variant (e.g. EX-L), null if none.
- "mileage" and "price" = integers, digits only (no $, commas, or the word miles).
- "title_status" = e.g. clean, salvage, rebuilt, lien; null if not stated.
- "seller_notes" = the seller's free-text description, trimmed; null if none.
Return ONLY the JSON object, no markdown, no commentary."""

FACT_KEYS = ["year", "make", "model", "trim", "mileage", "price", "title_status", "location", "seller_notes"]


def step_extract(image: Image.Image | None, text: str | None, trace: Trace) -> dict | None:
    start = time.time()
    prompt = EXTRACT_PROMPT if not text else f"{EXTRACT_PROMPT}\n\nListing text:\n{text}"
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image.convert("RGB")})
    content.append({"type": "text", "text": prompt})

    raw = _run_model(content, max_new_tokens=384)
    facts = _parse_json(raw)
    trace.add(
        "extract",
        {"has_image": image is not None, "text_len": len(text or "")},
        facts if facts is not None else {"_raw": raw},
        (time.time() - start) * 1000,
    )
    return facts


# --------------------------------------------------------------------------- #
# STEP 2 — VALIDATE / NORMALIZE (deterministic)
# --------------------------------------------------------------------------- #
def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = re.sub(r"[^\d]", "", value)
        return int(digits) if digits else None
    return None


def step_validate(facts: dict, trace: Trace) -> dict:
    start = time.time()
    clean = {key: facts.get(key) for key in FACT_KEYS}
    clean["mileage"] = _to_int(clean.get("mileage"))
    clean["price"] = _to_int(clean.get("price"))
    year = _to_int(clean.get("year"))
    clean["year"] = year if year and 1950 <= year <= 2027 else None
    for key in ("make", "model", "trim", "title_status", "location", "seller_notes"):
        val = clean.get(key)
        clean[key] = val.strip() if isinstance(val, str) and val.strip() else None
    trace.add("validate", facts, clean, (time.time() - start) * 1000)
    return clean


# --------------------------------------------------------------------------- #
# STEP 3 — RED-FLAG CHECK (deterministic rules — always fire, model-independent)
# --------------------------------------------------------------------------- #
TITLE_BAD = ("salvage", "rebuilt", "rebuild", "flood", "junk", "lemon", "lien", "branded")
SCAM_PATTERNS = [
    (r"\bdeposit\b", "Asks for a deposit (classic scam setup)."),
    (r"\b(wire|western union|gift card|zelle|escrow|paypal friends)\b", "Pushes an untraceable / unusual payment method."),
    (r"\b(ship|shipping|deliver|delivery)\b", "Offers to ship the car — common for cars that don't exist."),
    (r"\b(overseas|deployed|deployment|military|out of (the )?country|abroad)\b", "Seller claims to be away / overseas (can't meet)."),
    (r"\bno (test ?drive|inspection|meet)\b|can'?t (do|meet|test)", "Refuses a test drive or in-person meeting."),
    (r"\b(asap|urgent|today|this week|quick sale|need it gone)\b", "Manufactures urgency to rush the buyer."),
]
CONTRADICTION = (r"no (issues|problems|faults)", r"(accident|crash|repaired|fixed|damage)")


def step_red_flags(facts: dict, trace: Trace) -> list[dict]:
    start = time.time()
    flags: list[dict] = []
    notes = (facts.get("seller_notes") or "").lower()
    title = (facts.get("title_status") or "").lower()

    if any(word in title for word in TITLE_BAD):
        flags.append({"severity": "high", "issue": f"Title is '{facts['title_status']}', not clean — major value and insurance risk."})

    for pattern, message in SCAM_PATTERNS:
        if re.search(pattern, notes):
            flags.append({"severity": "high", "issue": message})

    if re.search(CONTRADICTION[0], notes) and re.search(CONTRADICTION[1], notes):
        flags.append({"severity": "medium", "issue": "Description contradicts itself ('no issues' but mentions an accident/repair)."})

    for key, label in (("price", "price"), ("mileage", "mileage"), ("title_status", "title status"), ("location", "location")):
        if facts.get(key) in (None, ""):
            flags.append({"severity": "low", "issue": f"Listing is missing the {label}."})

    trace.add("red_flags", {"title": title, "notes_len": len(notes)}, flags, (time.time() - start) * 1000)
    return flags


# --------------------------------------------------------------------------- #
# STEP 4 — PRICE SANITY (deterministic depreciation heuristic, NOT the model)
# --------------------------------------------------------------------------- #
# The model recalls market prices badly (it anchors near MSRP), so a bounded
# heuristic is safer than a confidently-wrong on-device band. It is rough on
# purpose — we label it clearly and link out to KBB/Edmunds for the real number.
THIS_YEAR = 2026
LUXURY = {"bmw", "mercedes", "mercedes-benz", "audi", "lexus", "porsche", "jaguar",
          "land rover", "cadillac", "infiniti", "acura", "volvo", "tesla"}
RELIABLE = {"toyota", "honda", "subaru", "mazda", "lexus"}
TRUCK_SUV = {"truck", "pickup", "suv", "f-150", "f150", "silverado", "ram", "tacoma",
             "tahoe", "suburban", "wrangler", "4runner"}


def _expected_value(facts: dict) -> dict | None:
    """Very rough private-party value midpoint + band. Bounded, transparent, label as an estimate."""
    year, make = facts.get("year"), (facts.get("make") or "").lower()
    if not isinstance(year, int) or not make:
        return None
    age = max(0, THIS_YEAR - year)
    base = 55_000 if make in LUXURY else 27_000
    blob = f"{make} {facts.get('model') or ''} {facts.get('trim') or ''}".lower()
    if any(t in blob for t in TRUCK_SUV):
        base = max(base, 40_000)
    retention = 0.86 if make in LUXURY else 0.92 if make in RELIABLE else 0.90
    value = base * (retention ** age)

    mileage = facts.get("mileage")
    if isinstance(mileage, int):
        expected_miles = 12_000 * max(age, 1)
        delta = (mileage - expected_miles) / 400_000  # ~$0.x/mi nudge, bounded
        value *= max(0.6, min(1.1, 1 - delta))
    if any(w in (facts.get("title_status") or "").lower() for w in TITLE_BAD):
        value *= 0.6  # branded titles trade well below clean comps

    value = max(800, value)
    return {"low": _round_to(int(value * 0.82), 100), "high": _round_to(int(value * 1.18), 100),
            "mid": int(value)}


def step_price(facts: dict, trace: Trace) -> dict:
    start = time.time()
    band = _expected_value(facts) or {}
    asking = facts.get("price")
    flag = None
    if band and isinstance(asking, int):
        ratio = asking / band["mid"]  # heuristic is noisy, so bands are wide
        if ratio < 0.55:
            band["verdict"] = "far below market"
            flag = {"severity": "high",
                    "issue": "Price is far below the typical market value for this car — a classic too-good-to-be-true / bait pattern."}
        elif ratio < 0.8:
            band["verdict"] = "below market"
        elif ratio <= 1.35:
            band["verdict"] = "in line with market"
        else:
            band["verdict"] = "above market"
    if facts.get("year") and facts.get("make") and facts.get("model"):
        query = f"{facts['year']} {facts['make']} {facts['model']}".replace(" ", "+")
        band["kbb_url"] = f"https://www.kbb.com/cars-for-sale/all?keyword={query}"
        band["edmunds_url"] = f"https://www.edmunds.com/inventory/srp.html?searchText={query}"
    trace.add("price", facts, {**band, "extra_flag": flag}, (time.time() - start) * 1000)
    band["_flag"] = flag
    return band


# --------------------------------------------------------------------------- #
# STEP 5 — ADVISE (deterministic: questions, walk-away price, summary)
# --------------------------------------------------------------------------- #
def _round_to(value: int, step: int = 100) -> int:
    return int(round(value / step) * step)


def step_advise(facts: dict, flags: list[dict], price: dict, trace: Trace) -> dict:
    start = time.time()
    notes = (facts.get("seller_notes") or "").lower()
    title = (facts.get("title_status") or "").lower()
    mileage, asking = facts.get("mileage"), facts.get("price")

    questions: list[str] = []
    if any(word in title for word in TITLE_BAD):
        questions.append(f"The title is '{facts['title_status']}' — what was the damage, and can I see the repair invoices and the insurance/accident report?")
    if re.search(CONTRADICTION[1], notes):
        questions.append("You mention an accident or repair — what exactly was damaged and who did the work?")
    if re.search(r"\b(deposit|escrow|gift card|wire|ship|shipping|overseas|deployed)\b", notes):
        questions.append("I only pay in person after seeing the car — can we meet locally with no deposit or shipping?")
    if re.search(r"no (test ?drive|inspection|meet)|can'?t (do|meet|test)", notes):
        questions.append("Can I take it for a test drive and have my own mechanic do a pre-purchase inspection?")
    if isinstance(mileage, int) and mileage >= 120_000:
        questions.append(f"At {mileage:,} miles, what major maintenance (timing belt, transmission service, brakes) has been done?")
    if asking in (None, ""):
        questions.append("What is your asking price, and is it firm?")
    if facts.get("location") in (None, ""):
        questions.append("Where is the car located, and where can we meet to see it?")

    for default in (
        "Do you have the title in hand, and is it in your name with no liens?",
        "Are there any warning lights or known mechanical issues right now?",
        "How many owners has it had, and do you have service records?",
        "Why are you selling it?",
    ):
        if len(questions) >= 5:
            break
        questions.append(default)
    questions = questions[:5]

    high_risk = any(f["severity"] == "high" for f in flags)
    med_risk = any(f["severity"] == "medium" for f in flags)
    # Anchor to the real asking price (known data); pull down for risk or clear overpricing.
    walk = None
    if isinstance(asking, int):
        base = asking * (0.8 if high_risk else 0.9 if med_risk else 0.97)
        if price.get("high") and asking > price["high"]:
            base = min(base, price["high"])  # don't endorse paying over a high estimate
        walk = _round_to(int(base))
    elif price.get("low"):
        walk = price["low"]

    highs = [f for f in flags if f["severity"] == "high"]
    if highs:
        summary = f"High-risk listing — {len(highs)} serious red flag(s), starting with: {highs[0]['issue']} Proceed only with an in-person inspection, if at all."
    elif med_risk:
        summary = "Some concerns worth clearing up before you commit — see the questions below, and inspect in person."
    else:
        summary = "No major red flags detected — looks worth a closer look. Still verify condition and paperwork in person."

    out = {"questions": questions, "walk_away_price": walk, "summary": summary}
    trace.add("advise", {"n_flags": len(flags), "price": price}, out, (time.time() - start) * 1000)
    return out


# --------------------------------------------------------------------------- #
# STEP 5 — ASSEMBLE verdict
# --------------------------------------------------------------------------- #
DISCLAIMER = (
    "_Estimate only — not financial or mechanical advice. AI-generated from a single "
    "listing on a ~1.3B on-device model; verify everything in person._"
)
SEV = {"high": "🔴", "medium": "🟠", "low": "🟡"}
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def _merge_flags(rule_flags: list[dict], model_flags: list[dict]) -> list[dict]:
    merged, seen = [], set()
    for flag in rule_flags + model_flags:
        issue = (flag.get("issue") or "").strip()
        key = issue.lower()[:60]
        if not issue or key in seen:
            continue
        seen.add(key)
        merged.append({"severity": (flag.get("severity") or "low").lower(), "issue": issue})
    merged.sort(key=lambda f: SEV_ORDER.get(f["severity"], 3))
    return merged


def _facts_table(facts: dict) -> str:
    labels = [("year", "Year"), ("make", "Make"), ("model", "Model"), ("trim", "Trim"),
              ("mileage", "Mileage"), ("price", "Price"), ("title_status", "Title"),
              ("location", "Location")]
    rows = ["| Field | Value |", "| --- | --- |"]
    for key, label in labels:
        val = facts.get(key)
        if key == "mileage" and isinstance(val, int):
            val = f"{val:,} mi"
        elif key == "price" and isinstance(val, int):
            val = f"${val:,}"
        rows.append(f"| {label} | {val if val not in (None, '') else '—'} |")
    return "\n".join(rows)


def _render(facts: dict, flags: list[dict], reason: dict, trace: Trace) -> str:
    parts = ["## 🚗 Lot Scout — verdict"]
    if reason.get("summary"):
        parts.append(f"> {reason['summary']}")

    parts.append("### Facts\n" + _facts_table(facts))

    parts.append("### 🚩 Red flags")
    if flags:
        parts.append("\n".join(f"- {SEV.get(f['severity'], '🟡')} {f['issue']}" for f in flags))
    else:
        parts.append("_None detected — still verify in person._")

    fp = reason.get("fair_price") or {}
    low, high = fp.get("low"), fp.get("high")
    parts.append("### 💰 Price check")
    if isinstance(low, int) and isinstance(high, int):
        line = f"Rough on-device ballpark: **${low:,} – ${high:,}**"
        if fp.get("verdict"):
            line += f" — asking price looks **{fp['verdict']}**."
        parts.append(line)
        parts.append("_Very rough heuristic, not a valuation — confirm the real number before you negotiate._")
    else:
        parts.append("_Not enough info for a ballpark — confirm the market value below._")
    links = [f"[{name}]({fp[key]})" for name, key in (("KBB", "kbb_url"), ("Edmunds", "edmunds_url")) if fp.get(key)]
    if links:
        parts.append("Check real market value: " + " · ".join(links))

    questions = reason.get("questions") or []
    if questions:
        parts.append("### ❓ Ask the seller\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions[:5], 1)))

    walk = reason.get("walk_away_price")
    if isinstance(walk, int):
        parts.append(f"### 🚪 Suggested ceiling\n**${walk:,}** — a sensible maximum to pay; start your offer below it.")

    parts.append(f"<sub>pipeline: {' → '.join(s['step'] for s in trace.steps)} · {trace.total_ms:.0f} ms</sub>")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze(image: Image.Image | None, text: str | None) -> tuple[str, list]:
    if image is None and not (text and text.strip()):
        return "⚠️ Upload a listing screenshot or paste the listing text to get started.", []

    trace = Trace()
    facts = step_extract(image, text, trace)
    if facts is None:
        raw = trace.steps[-1]["output"].get("_raw", "")
        return ("## 🚗 Lot Scout\n\nCouldn't read structured facts from that input. "
                f"Raw model output:\n\n```\n{raw}\n```\n\n" + DISCLAIMER), trace.steps

    facts = step_validate(facts, trace)
    if not any(facts.get(k) for k in ("make", "model", "price", "year", "mileage")):
        return ("## 🚗 Lot Scout\n\nI couldn't find a used-car listing in that input — no make, "
                "model, price, year, or mileage. Try a clearer listing screenshot or paste the "
                "listing text.\n\n" + DISCLAIMER), trace.steps
    rule_flags = step_red_flags(facts, trace)
    price = step_price(facts, trace)
    extra = [price["_flag"]] if price.get("_flag") else []
    flags = _merge_flags(rule_flags, extra)  # dedupe + sort by severity
    advice = step_advise(facts, flags, price, trace)
    reason = {"fair_price": price, **advice}  # shape expected by _render
    return _render(facts, flags, reason, trace), trace.steps


def _trace_file(steps: list) -> str | None:
    """Write the run's pipeline trace to a downloadable JSON (Open Trace)."""
    if not steps:
        return None
    payload = {"app": "lot-scout", "model": MODEL_ID, "pipeline": [s["step"] for s in steps], "steps": steps}
    handle = tempfile.NamedTemporaryFile(mode="w", suffix="_lot-scout-trace.json", delete=False, encoding="utf-8")
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.close()
    return handle.name


@spaces.GPU(duration=GPU_DURATION)
def analyze_gpu(image: Image.Image | None, text: str | None) -> tuple[str, str | None]:
    markdown, steps = analyze(image, text)
    return markdown, _trace_file(steps)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
EXAMPLES = [
    ["assets/examples/listing_accord.png", ""],
    ["assets/examples/listing_camry.png", ""],
    ["assets/examples/listing_bmw.png", ""],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Lot Scout", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🚗 Lot Scout\n"
            "Paste a used-car listing or drop a screenshot. Lot Scout reads it **locally** on "
            "MiniCPM-V 4.6 and returns the facts, red flags, a fair-price estimate, and the "
            "questions to ask before you waste a Saturday. _Runs entirely on-device — no cloud APIs._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Listing screenshot", height=340)
                text_in = gr.Textbox(label="…or paste listing text", lines=6,
                                     placeholder="2013 Honda Accord, 162k miles, $7,200, rebuilt title…")
                run_btn = gr.Button("Analyze listing", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=[image_in, text_in], label="Try an example")
            with gr.Column(scale=1):
                verdict = gr.Markdown("Your verdict will appear here.")
                trace_file = gr.File(label="⬇️ Agent trace (JSON) — Open Trace", interactive=False)

        run_btn.click(analyze_gpu, inputs=[image_in, text_in], outputs=[verdict, trace_file])
    return demo


if __name__ == "__main__":
    build_ui().launch()
