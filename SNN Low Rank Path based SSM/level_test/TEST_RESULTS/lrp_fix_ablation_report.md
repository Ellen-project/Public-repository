## LRP Fix Ablation

| Mode | Eval Loss | Eval PPL | Mean Active | Logits L2 vs Zero | Path/Base Ratio |
| --- | --- | --- | --- | --- | --- |
| zero_gate | 10.1405 | 25349.77 | 0.0000 | 0.000000 | 0.001266 |
| cached_gate | 10.0933 | 24180.61 | 1.1572 | 0.071065 | 0.044220 |
| all_on_gate | 10.0303 | 22704.15 | 4.0000 | 0.164013 | 0.090838 |
| random_gate_same_density | 10.0961 | 24249.09 | 1.1353 | 0.070057 | 0.044180 |
| force_min_active_1 | 10.0682 | 23581.44 | 1.1572 | 0.105524 | 0.066584 |
| force_min_active_2 | 10.0419 | 22969.46 | 1.1572 | 0.144542 | 0.083407 |

Verdicts: PASS