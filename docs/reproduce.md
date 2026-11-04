# Reproduce the paper

All runs need the planner server up (`bash scripts/serve_planner.sh`), except LIBERO-Displace,
which runs fixed scripted plans. Every script resumes: run the same command again to finish or retry
failed items.

## LIBERO-PRO

Mean success (%) over 10 tasks x 50 trials per cell.

| Run | Object pos / task | Goal pos / task | Spatial pos / task | Mean | Command |
|---|---|---|---|---|---|
| Table 1 | 86.4 / 82.4 | 71.6 / 61.8 | 65.4 / 83.0 | 75.1 | `bash scripts/repro_libero_pro.sh table1` |
| Strict | 77.8 / 76.6 | 64.0 / 42.8 | 62.4 / 81.6 | 67.5 | `bash scripts/repro_libero_pro.sh strict` |

| Run | Object lang / obj | Goal lang / obj | Spatial lang / obj | Mean | Command |
|---|---|---|---|---|---|
| Held out | 92.4 / 77.4 | 64.6 / 47.2 | 50.8 / 85.8 | 69.7 | `bash scripts/repro_libero_pro.sh heldout` |

Summaries:

```bash
python -m spark_bench.summarize results/libero_pro/table1 results/libero_pro/heldout --cells --ci --suite-means
python -m spark_bench.summarize results/libero_pro/strict --timing   # time per trial
```

Each run is one item list in `repro/` (600 lines, 5 trials per line):

    suite perturbation task_id trial_offset n_trials arm

## LIBERO-Displace

Successes out of 45 at 0 / 2 / 5 / 10 cm, verification and recovery on vs off.

| Run | On | Off | Command (`on` arm, then `off` arm) |
|---|---|---|---|
| 40-step slide, success check after each attempt | 39 / 37 / 35 / 30 | 33 / 26 / 7 / 1 | `bash scripts/repro_dyn.sh displace on results/dyn_slide`, `... displace off results/dyn_slide` |
| 40-step slide, strict | 36 / 28 / 15 / 12 | 33 / 25 / 7 / 1 | `... displace onec_st results/dyn_slide_strict`, `... displace off_st ...` |
| Instantaneous, strict | 33 / 25 / 12 / 5 | 33 / 26 / 2 / 0 | `... displace onec_st_inst results/dyn_inst_strict`, `... displace off_st_inst ...` |

Print the table from a run with `python -m spark_bench.dyn.metrics <out_root>`.

## Arms

`scripts/run_item.sh` maps the arm token at the end of each item to flags:

| Token | Effect |
|---|---|
| `wv` | wrist camera checks each grasp (`--wrist-verify`) |
| `rb` | spatial relations bind to the right object instance (`--relation-bind`) |
| `_or` | the success check after each attempt decides whether to retry (`SPARK_ORACLE_RETRY=1`) |
| `_ph` | detection phrase table (`SPARK_PHRASES_V2=1`) |
| `_rgs` | wrist camera re-grounds the pick target, at most 5 cm (`SPARK_PICK_REGROUND=1`, `SPARK_PICK_REGROUND_GATE_M=0.05`) |

Without `_or` the run is strict: only the robot's own checks (camera scene diff, empty or slipped
grasp) start a recovery.

## Protocol

- Trial i loads LIBERO-PRO initial state i and never resets after loading, as in LIBERO's own evaluation.
- The planner sees the instruction and the camera images only, never simulator state or the task's
  BDDL file. Every trial plans from scratch; a planner error fails the trial.
- The phrase table and the 5 cm gate were tuned on separate placements (environment seeds 64 to 99),
  never on the evaluation states.
- LIBERO-Displace runs fixed plans, so the arms differ only in verification and recovery. Its runner
  resets after loading to sample new placements.
