# python
from pathlib import Path

from datasets import load_dataset
import json
import re
from itertools import islice

ds = load_dataset(
    "codeparrot/xlcost-text-to-code",
    "C++-program-level",
    split="train",
    streaming=True,
    trust_remote_code=True
)

def extract_items_from_text(text: str):
    parts = text.split("|", 1)
    after = parts[1] if len(parts) > 1 else parts[0]

    pieces = [p.strip() for p in after.split(";") if p.strip()]

    # drop leading "C ++ Program to implement..." (case-insensitive)
    pieces.pop(0)

    # truncate at the first occurrence of "Driver code" (case-insensitive),
    # removing it and everything after
    for i, p in enumerate(pieces):
        if re.search(r'(?i)driver\s*code', p):
            pieces = pieces[:i]
            break

    return pieces
script_dir = Path(__file__).parent.parent.absolute()

out_path = script_dir / "data/first1000.jsonl"
with open(out_path, "w", encoding="utf-8") as fout:
    for record in islice(ds, 1000):
        code_text = record.get("code", "")
        text = record.get("text", "")
        pieces = extract_items_from_text(text)
        positive = " ; ".join(pieces)
        fout.write(json.dumps({"code": code_text, "positive": positive}, ensure_ascii=False) + "\n")
