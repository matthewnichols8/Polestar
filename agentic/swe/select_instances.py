"""Pick the SWE-bench Verified instances for the DiffusionGemma runs (fixed seed) and write instances.txt + filter.txt."""
import random, re
from datasets import load_dataset

N, SEED = 20, 42
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
ids = sorted(ds["instance_id"])
picked = sorted(random.Random(SEED).sample(ids, N))
open("instances.txt", "w").write("\n".join(picked) + "\n")
open("filter.txt", "w").write("^(" + "|".join(re.escape(i) for i in picked) + ")$\n")
by_id = {r["instance_id"]: r for r in ds}
for i in picked:
    print(i, by_id[i]["difficulty"] if "difficulty" in by_id[i] else "")
