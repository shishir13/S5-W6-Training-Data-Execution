# Training Data Execution System

A complete, auditable training data pipeline for large language model pre-training. Built for ERA V5 Session 6.

## One command

```bash
uv sync
uv run python run_demo.py
```

## What it demonstrates

| Requirement | How |
|-------------|-----|
| Immutable shards + manifests | SHA-256 hash locked at creation; verified on every load |
| Frozen tokenizer | GPT-2 vocab hash computed once and stored in every manifest |
| Packing policies | TEXT, CODE, INSTRUCTION, EVAL — each with correct loss\_mask and position IDs |
| Position ID reset | Resets to 0 at every EOS token boundary (multi-doc packing correctness) |
| Curriculum schedule | 4 phases (warmup → foundation → skill\_ramp → anneal) over 60 steps |
| Protected floors | 4 lanes with minimum realized-share floors enforced by OPUS |
| OPUS decisions | accept / reject / defer / floor\_override logged per batch |
| Consumption ledger | JSONL, append-only, byte-offset tracked for crash-safe truncation |
| Learning ledger | Per-batch loss, perplexity, loss-bearing tokens, linked to consumption |
| Checkpoint | Model + optimizer + all RNG states + ledger byte offsets + pre-computed next\_batch\_id |
| Deliberate crash | Raises SystemExit at step 25; writes crash.flag |
| Crash recovery | Loads checkpoint, truncates ledgers to exact byte offsets, verifies next batch\_id |
| Replay | Reloads shard tokens from ledger spans, recomputes batch\_ids, asserts match |
| Fork | Loads step-10 checkpoint with different OPUS seed, runs 5 steps, confirms divergence |
| Eval firewall | Eval shards never enter candidate buffer; loss\_mask=0 even if they did |
| Throughput report | tokens/sec, loss-bearing tokens/sec, packing utilization % |

## Architecture

```
documents
  └─ corpus.py          synthetic docs (5 lanes x 5 docs + 3 eval)
       └─ tokenizer.py  FrozenTokenizer (GPT-2, hash-locked)
            └─ sharding.py   .bin shards + SHA-256 content hash
                 └─ manifest.py   ShardManifest dataclass + file validation
                      └─ mixture.py    curriculum phases + lane weights
                           └─ opus.py       OPUS selector (accept/reject/defer/floor_override)
                                └─ packing.py    TEXT/CODE/INSTRUCTION/EVAL policies + batch_id
                                     └─ model.py     TinyGPT (2-layer, 128-dim, ~6.86M params)
                                          └─ trainer.py   masked CE loss + AdamW
                                               └─ ledger.py    consumption + learning JSONL ledgers
                                                    └─ checkpoint.py  save/load + next_expected_batch_id
                                                         └─ recovery.py   crash detect + ledger truncation
                                                              └─ replay.py     batch_id reconstruction
                                                                   └─ audit.py      evidence bundle
                                                                        └─ perf.py       throughput
```

## Key design decisions

**batch\_id = sha256(step\_bytes ‖ token\_bytes)**  
A pure function of content — not wall time. Enables replay and resume verification without rerunning training logic.

**JSONL ledger with byte offsets**  
Each entry records its own byte offset. On crash recovery, ledgers are truncated to the exact offset stored in the checkpoint — no line-count ambiguity, no SQLite.

**next\_expected\_batch\_id pre-computed at checkpoint save time**  
Computed before the crash happens. On resume, the produced batch\_id is asserted against this pre-stored value. A buggy resume cannot forge a match.

**Position IDs reset at EOS**  
Each packed document starts at position 0 regardless of where it sits in the context window. Matches GPT-NeoX / MPT / Falcon practice.

**DEFER decision**  
Low-utility candidates from floored lanes are held one step rather than rejected, reducing floor-override oscillation.

## Running tests

```bash
uv run pytest tests/ -v
```

## Generated artifacts

```
submission_artifacts/
  run.log           complete event log
  evidence.json     machine-readable PASS/FAIL for every requirement
  evidence.md       human-readable summary table
  performance.json  throughput and packing metrics
  manifests/        one .json per shard
  shards/           one .bin per document
  ledgers/          consumption_ledger.jsonl  learning_ledger.jsonl
  checkpoints/      ckpt_NNNNN_coffM.pt files
```

## Model

TinyGPT: 2 layers, 4 heads, d\_model=128, d\_ff=512, vocab=50257 (GPT-2).  
~6.86M parameters. Runs on CPU. 60 steps complete in under 3 minutes.
