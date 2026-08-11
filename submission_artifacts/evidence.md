# Evidence Summary -- Training Data Execution System

Run ID: `1eeade64`  |  Steps: 60  |  All Pass: **YES**

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Tokenizer integrity | **PASS** | checked 23 manifests |
| Evaluation firewall | **PASS** | eval shards in consumption ledger: 0 |
| Packing correctness | **PASS** | loss_mask=0 for eval/prompt; position_ids reset at EOS; batch_id deterministic |
| Mixture compliance | **PASS** | planned vs actual shares: {"english_web": 0.2833, "code": 0.2667, "math": 0.2333, "instruction": 0.2167} |
| OPUS audit trail | **PASS** | {"floor_override": 3, "accept": 41, "reject": 15, "defer": 1} |
| Crash recovery | **PASS** | expected=acb3c2108abe826c produced=acb3c2108abe826c |
| Replay hash verification | **PASS** | matched=60/60 |
| Learning trace | **PASS** | loss linked to source data: 60 learning entries, 60 linked to consumption ledger |
| Throughput | **PASS** | tokens/sec=3025.1, loss_tokens/sec=2774.7 |
| Shard content hashes | **PASS** | 23/23 shards verified |
| Loss mask / eval exclusion | **PASS** | no eval entries in consumption ledger |
| Protected floor invariants | **PASS** | {"english_web": {"floor": 0.16, "realized": 0.2833, "ok": true}, "code": {"floor": 0.07, "realized": 0.2667, "ok": true} |
| Ledger completeness | **PASS** | consumption entries: 60, learning entries: 60, unmatched: 0 |
| Fork from earlier checkpoint | **PASS** | fork_step=10 original=ebd5e4d366384f55 forked=2fff0483e9e6fa46 |
| Packing utilization | **PASS** | mean=51.8% |
| Position ID EOS reset | **PASS** | verified by test_packing.py::test_position_ids_reset_at_eos |

## Ledger Summary

- Consumption entries: 60
- Learning entries: 60
- Total loss-bearing tokens: 14012
- Mean loss: 32.8664
- Mean perplexity: 3.023490750375575e+36

## Performance

- total_steps: 60
- total_tokens: 15277
- total_loss_bearing_tokens: 14012
- total_time_sec: 5.05
- tokens_per_sec: 3025.1
- loss_tokens_per_sec: 2774.7
- mean_packing_utilization_pct: 51.75
- min_packing_utilization_pct: 17.58
- max_packing_utilization_pct: 100.0