# Evidence Summary -- Training Data Execution System

Run ID: `0940cae7`  |  Steps: 59  |  All Pass: **YES**

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Tokenizer integrity | **PASS** | checked 23 manifests |
| Evaluation firewall | **PASS** | eval shards in consumption ledger: 0 |
| Packing correctness | **PASS** | loss_mask=0 for eval/prompt; position_ids reset at EOS; batch_id deterministic |
| Mixture compliance | **PASS** | planned vs actual shares: {"english_web": 0.3559, "code": 0.3051, "math": 0.1695, "instruction": 0.1695} |
| OPUS audit trail | **PASS** | {"floor_override": 3, "accept": 36, "reject": 20} |
| Crash recovery | **PASS** | expected=b0faf96ff3df17ad produced=b0faf96ff3df17ad |
| Replay hash verification | **PASS** | matched=59/59 |
| Learning trace | **PASS** | loss linked to source data: 59 learning entries, 59 linked to consumption ledger |
| Throughput | **PASS** | tokens/sec=11648.3, loss_tokens/sec=10623.1 |
| Shard content hashes | **PASS** | 23/23 shards verified |
| Loss mask / eval exclusion | **PASS** | no eval entries in consumption ledger |
| Protected floor invariants | **PASS** | {"english_web": {"floor": 0.16, "realized": 0.3559, "ok": true}, "code": {"floor": 0.07, "realized": 0.3051, "ok": true} |
| Ledger completeness | **PASS** | consumption entries: 59, learning entries: 59, unmatched: 0 |
| Fork from earlier checkpoint | **PASS** | fork_step=10 original=ebd5e4d366384f55 forked=2fff0483e9e6fa46 |
| Packing utilization | **PASS** | mean=46.6% |
| Position ID EOS reset | **PASS** | verified by test_packing.py::test_position_ids_reset_at_eos |

## Ledger Summary

- Consumption entries: 59
- Learning entries: 59
- Total loss-bearing tokens: 13077
- Mean loss: 32.6102
- Mean perplexity: 1.0432164299846673e+35

## Performance

- total_steps: 59
- total_tokens: 14339
- total_loss_bearing_tokens: 13077
- total_time_sec: 1.231
- tokens_per_sec: 11648.3
- loss_tokens_per_sec: 10623.1
- mean_packing_utilization_pct: 46.56
- min_packing_utilization_pct: 19.53
- max_packing_utilization_pct: 69.53