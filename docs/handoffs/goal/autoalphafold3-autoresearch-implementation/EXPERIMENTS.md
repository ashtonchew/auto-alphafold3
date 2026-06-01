# Autoresearch Experiments

No autoresearch experiments have been run for this implementation wave.

## Deterministic Ladder Reserved IDs

| Trial | Purpose | Status |
| --- | --- | --- |
| `T120` | short training baseline smoke, 10 steps | planned |
| `T121` | first geometry patch smoke, 10 steps | planned |
| `T122` | short training baseline trial, 250 steps | planned |
| `T123` | best geometry patch trial, 250 steps | planned |
| `T124` | no geometry auxiliary ablation, 250 steps | planned |
| `T125` | sampler after best checkpoint, inference only | planned |

## Local Fixture Evidence

PR 2 adds a fixture-backed 2-step short-training test. It runs on the local
synthetic NanoFold fixture, writes trial-scoped artifacts under pytest
temporary directories, and stamps `official_benchmark_result=false`.

## Live Actions

No live Modal or open-ended search action is approved in PR 1.
