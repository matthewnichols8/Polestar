# Agentic DiffusionGemma + Polestar (τ²-bench, SWE-bench)

DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) served through dInfer with Polestar CACHE / COMMIT behind an OpenAI-compatible API, and run as a tool-calling agent on τ²-bench (retail) and SWE-bench Verified (mini-swe-agent). Developed on godchang (2× TITAN RTX), Nov 2026.

## Start here

| Doc | What |
|---|---|
| [AGENTIC_DGEMMA_POLESTAR.md](AGENTIC_DGEMMA_POLESTAR.md) | How the stack works, step by step |
| [results/tau2/TAU2_RETAIL_RESULTS.md](results/tau2/TAU2_RETAIL_RESULTS.md) | τ²-bench retail: baseline 13/20, CACHE 16/20, full Polestar 18/20 (tasks 0–19) |
| [results/swe/SWE_RESULTS.md](results/swe/SWE_RESULTS.md) | SWE-bench Verified, 20 tasks: baseline 3/20, CACHE 4/20, full Polestar (18k cap) 0/20 |
| [PLAN.md](PLAN.md), `results/PHASE0_RESULTS.md`, `results/PHASE1_RESULTS.md` | Original plan and early tool-call probes |

## Layout

| Path | What |
|---|---|
| `server/` | `serve_openai.py` (OpenAI-compatible server, prefix-KV reuse, `--polestar`, `--polestar-commit`), `gemma_format.py` (OpenAI ⇄ Gemma tool syntax), tests |
| `swe/` | SWE-bench without Docker: `fakesingularity` (crane + bwrap), `eval_bwrap.py` (local scorer), `run_all.sh`, `dgemma_swebench.yaml`, the 20 task ids |
| `probes/`, `evidence/` | Tool-call probes; Polestar CACHE trace evidence |
| `results/` | Write-ups, per-run `config.json`, scores and summaries. Full SWE trajectories and server logs are not committed (74 MB) and stay on godchang under `/scratch/mi72/agentic-dgemma/results/swe/` |
| `requirements-*.txt` | `pip freeze` of the three environments |

No secrets are committed. τ²-bench's user simulator needs an OpenAI key; put it in a `chmod 600` env file outside the repo.

## Set up on a new server

1. **Clone** into a directory with a lot of free space, e.g. `~/USERSCRATCH` (the model weights alone are about 50 GB):
   ```bash
   git clone -b feature/agentic-dgemma git@github.com:matthewnichols8/Polestar.git && cd Polestar
   ```
2. **Fix paths.** The code was written on godchang, where this folder was `/scratch/mi72/agentic-dgemma` and the repo `/scratch/mi72/Polestar`. From the repo root, point every hard-coded path at the new location (`BASE` holds the envs, HF cache, tmp and rootfs cache):
   ```bash
   BASE=$(dirname $PWD)
   grep -rlZ /scratch/mi72 agentic/server agentic/swe agentic/probes | xargs -0 sed -i \
     -e "s#/scratch/mi72/agentic-dgemma#$PWD/agentic#g" -e "s#/scratch/mi72/Polestar#$PWD#g" -e "s#/scratch/mi72#$BASE#g"
   ```
   The paths cover the environments (`/scratch/mi72/envs/{polestar,swe,tau2}`), `TMPDIR=/scratch/mi72/tmp`, `HF_HOME=/scratch/mi72/hf` and the rootfs cache `/scratch/mi72/swe-rootfs`.
3. **Server environment** (Python 3.10). Pinned: `torch==2.7.1+cu126`, `transformers==4.49.0`. Both support the H200 (sm_90).
   ```bash
   python3.10 -m venv $BASE/envs/polestar   # or conda
   $BASE/envs/polestar/bin/pip install -r agentic/requirements-polestar.txt --extra-index-url https://download.pytorch.org/whl/cu126
   $BASE/envs/polestar/bin/pip install -e . -e integrations/dinfer
   ```
4. **SWE-bench environment** (Python 3.12): `pip install -r requirements-swe.txt`, which includes mini-swe-agent 2.4.6, swebench 5.0.2 and litellm.
   - Download `crane` 0.22.1 (the go-containerregistry release binary) into `agentic/swe/bin/crane`.
   - The sandbox needs `bwrap` (bubblewrap) with unprivileged user namespaces. Test with `bwrap --unshare-user --uid 0 --bind / / id`. If it fails, check whether the cluster offers Apptainer/Singularity, which mini-swe-agent supports directly: point `executable` in `dgemma_swebench.yaml` at it instead of `fakesingularity`.
   - The first run fetches about 20 images into the rootfs cache.
5. **τ²-bench environment** (Python 3.12): `pip install -r requirements-tau2.txt`, plus `pip install -e` of `sierra-research/tau2-bench` @ `19f13ea`.
6. **Model weights:**
   ```bash
   HF_HOME=<dir>/hf huggingface-cli download google/diffusiongemma-26B-A4B-it
   ```
   Check that the compute node has internet first: `curl -I https://huggingface.co`. If it doesn't, download on the login node.
7. **Check that it works:**
   - `$BASE/envs/polestar/bin/python -m pytest agentic/server/test_gemma_format.py` (11 tests).
   - `python eval_bwrap.py --gold ...` should resolve 20/20, and `--empty ...` 0/20.
   - A one-task smoke run (see `results/swe/SWE_RESULTS.md`).

## Changes for a single 96 GB GPU

- **One GPU:** pass `--gpus 0` (all experts on one device). This also removes the two-GPU hang, which happened when only one GPU ran out of memory.
- **Bigger context:** raise `--max-prompt-tokens`, and `max_input_tokens` in `swe/model_registry.json`, e.g. to 32k or 64k. Probe memory first with `swe/context_probe.py`, because COMMIT uses the most memory.
- **Expert precision:** `--expert-dtype bf16` may fit and avoid int8 dequantisation. Compare against int8 for speed and quality.
- **Comparable reruns:** for a fair comparison of full Polestar, rerun all three configurations at the same cap, with `MAXP=<cap> bash swe/run_all.sh`.
- **Rogues Gallery limits:** `salloc` sessions are limited (e.g. 8 h). The three runs take about 3–4 h each on the TITANs, probably less on an H200. Submit each configuration as its own `sbatch` job rather than running everything in one interactive session.
