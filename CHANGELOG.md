# Changelog

## v2.0 — October 2026 (revision of PRLETTERS-D-25-01016)

* The implementation now follows Section 3 of the manuscript exactly: LMS-TRD is
  built from the NeRD-Rain architecture by the eight modifications of the ablation
  study, and every row M0–M8 is a configuration of one model class
  (`models/lms_trd.py`, `configs/ablation/`).
* M0 and M1 are verified to be identical to the official NeRD-Rain and
  NeRD-Rain-S (parameter counts and outputs; `tools/verify_nerd_rain.py`).
* The previous backbones (`unet_sst.py`, `sst_flat.py`) and their notes
  (`RECONCILIATION.md`, `docs/paper_reference.md`) were removed: they did not
  correspond to the architecture described in Section 3.
* All parameter and MAC counts are measured by `tools/profile_complexity.py`
  (results in `results/complexity.md`).
* Evaluation follows the Rain13K protocol (Y channel, 8-bit outputs).
* Added: latency benchmark protocol, ONNX export and AI Hub profiling script,
  activation checkpointing and gradient accumulation, per-run logs
  (`run.json`, `log.csv`), result aggregation, smoke test.

## v1.0 — initial release
