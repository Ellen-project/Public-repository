# Neuromorphic AI

Hybrid sequence-modeling experiments inspired by spiking neurons, synapses, and selective state-space models.

## Layout

- `src/`: importable model and data code
- `src/lrp_ssm/`: low-rank path SSM core, SNN router bridge, gate cache helpers
- `src/snn/`: pyramidal-neuron SNN simulator
- `src/lm/`: C4 language-model wrapper and C4 data utilities
- `scripts/`: runnable training, evaluation, calibration, cache, and smoke-test entry points
- `experiments/`: baseline comparisons, ablations, benchmarks, and generated reports

## Quick Check

Install dependencies first:

```powershell
pip install -r requirements.txt
```

Then run:

```powershell
python scripts/smoke_cpu.py
python scripts/train_ssm_smoke.py --epochs 1 --num-samples 4 --seq-len 4
```
