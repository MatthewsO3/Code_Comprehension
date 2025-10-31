# create_distractors.py
from pathlib import Path
from datasets import load_dataset
import json
import re
from itertools import islice

# This is the same dataset used in your other scripts
ds = load_dataset(
    "codeparrot/xlcost-text-to-code",
    "C++-program-level",
    split="train",
    streaming=True,
    trust_remote_code=True
)

script_dir = Path(__file__).parent.parent.absolute()

# NOTE: We are creating a new file here
out_path = script_dir / "data/distractors.jsonl"
print(f"Creating distractor file at: {out_path}")

# We take 50,000 records, starting AFTER your eval set (which ended at 2100)
# This ensures no overlap with your test queries.
# (2100 to 52100 = 50,000 records)
with open(out_path, "w", encoding="utf-8") as fout:
    # We only need the code for distractors
    for idx, record in enumerate(islice(ds, 8650, 9797)):
        code_text = record.get("code", "")

        # We need a unique URL/ID for each distractor
        distractor_url = f"distractor_{idx}"

        fout.write(json.dumps({"code": code_text, "url": distractor_url}, ensure_ascii=False) + "\n")

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} distractors...")

print(f"Finished creating {out_path} with 50,000 distractors.")
