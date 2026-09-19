import re, json
from langchain_core.tools import tool

# name: (low, high, meaning in plain words, specialist)
RANGES = {
    "hemoglobin":        (13.0, 17.0, "the protein in red blood cells that carries oxygen", "Hematologist"),
    "fasting glucose":   (70, 100, "your blood sugar level when you haven't eaten", "Endocrinologist"),
    "total cholesterol": (0, 200, "the total fat-like substance in your blood", "Cardiologist"),
    "creatinine":        (0.7, 1.3, "a waste product that shows how well kidneys work", "Nephrologist"),
    "tsh":               (0.4, 4.0, "a hormone that controls your thyroid gland", "Endocrinologist"),
}
FOLLOW_UP_WORDS = ["repeat", "follow up", "follow-up", "return if", "review in"]

def analyze(text: str) -> list[dict]:
    results = []
    for name, (low, high, meaning, doc) in RANGES.items():
        m = re.search(rf"{name}[^\d\n]{{0,20}}(\d+\.?\d*)", text, re.I)
        if not m:
            continue
        val = float(m.group(1))
        status = "LOW" if val < low else "HIGH" if val > high else "NORMAL"
        results.append({"test": name, "value": val, "range": f"{low}-{high}",
                        "status": status, "meaning": meaning, "specialist": doc})
    return results

def follow_up_lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines()
            if any(w in l.lower() for w in FOLLOW_UP_WORDS)]

@tool
def simplify_and_flag(record_text: str) -> str:
    """Rewrite lab values in plain language and flag any value outside its
    normal range or any follow-up instruction. Input: full text of the record."""
    lines, flags = [], []
    for r in analyze(record_text):
        icon = "OK" if r["status"] == "NORMAL" else "FLAG"
        lines.append(f"[{icon}] {r['test'].title()}: {r['value']} "
                     f"(normal {r['range']}) - {r['meaning']}. Status: {r['status']}.")
        if r["status"] != "NORMAL":
            flags.append(r["test"])
    for l in follow_up_lines(record_text):
        lines.append(f"[FOLLOW-UP] {l}")
    return json.dumps({"summary": lines, "flagged_tests": flags})

@tool
def find_specialist(flagged_tests: str) -> str:
    """Given comma-separated flagged test names, return which doctor or
    department to see for each. Example input: 'tsh, hemoglobin'."""
    out = []
    for t in [x.strip().lower() for x in flagged_tests.split(",") if x.strip()]:
        if t in RANGES:
            out.append(f"{t.title()} -> see a {RANGES[t][3]}")
    return "\n".join(out) or "No specialist needed."