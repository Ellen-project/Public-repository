# Low Rank Path based SSM vs Attention Baselines

## 1. 실험 목적

- Low Rank Path based SSM과 attention 계열 baseline의 language modeling 성능 및 효율 비교.
- C4 cache 기반 동일 데이터 조건에서 비교.

## 2. 실험 환경

| 항목 | 값 |
| --- | --- |
| Python | 3.12.9 |
| PyTorch | 2.12.0.dev20260228+cu128 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| device | cuda |
| AMP | False |
| seed | 1 |
| token cache | c4_token_cache_medium.pt |
| gate cache | c4_gate_cache_medium_dim64_calibrated.pt |

## 3. 비교 모델

| Model | Type | Params | Uses Gate | Complexity | Notes |
| --- | --- | --- | --- | --- | --- |
| lrp_ssm_fixed_calibrated | Low-rank path SSM fixed calibrated | 6475843 | yes | O(T*paths*rank) | cached/learned gates |
| lrp_ssm_learned_router | Low-rank path SSM learned router | 6484679 | yes | O(T*paths*rank) | cached/learned gates |
| transformer | Full causal attention | 6587072 | no | O(T^2) |  |
| linear_attention | Kernelized linear attention | 6586496 | no | O(T), simplified ELU+1 prefix | simplified baseline |
| local_attention | Sliding-window attention | 6587072 | no | O(T*w), w=32 |  |
| gru | Recurrent | 6507776 | no | O(T) recurrent |  |

## 4. 학습 설정

| 항목 | 값 |
| --- | --- |
| block_size | 64 |
| batch_size | 4 |
| max_train_steps | 50 |
| optimizer | AdamW |
| lr | 0.0003 |
| weight_decay | 0.0100 |
| grad_clip | 1.0000 |
| model_dim | 64 |
| num_layers | 3 |
| num_heads | 4 |
| state_dim | 64 |
| rank | 2 |
| num_paths | 4 |

## 5. 결과 요약

| Model | Params | Best Eval Loss | Best Eval PPL | Final Train Loss | Avg Train Tokens/s | Avg Eval Tokens/s | Peak GPU Memory MB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lrp_ssm_fixed_calibrated | 6475843 | 10.2434 | 28096.9531 | 10.1392 | 464.9307 | 787.4400 | 364.7549 | pass |
| lrp_ssm_learned_router | 6484679 | 10.2003 | 26910.8850 | 10.0699 | 548.6990 | 1231.8175 | 365.9155 | pass |
| transformer | 6587072 | 10.0036 | 22105.4477 | 9.8854 | 22256.8374 | 58928.2067 | 366.5283 | pass |
| linear_attention | 6586496 | 10.0548 | 23267.6413 | 9.9550 | 19797.3822 | 57443.1318 | 366.5166 | pass |
| local_attention | 6587072 | 10.0126 | 22305.1893 | 9.8933 | 22285.7224 | 61847.7959 | 366.5283 | pass |
| gru | 6507776 | 10.0714 | 23657.2020 | 9.9825 | 40123.2405 | 129289.6518 | 365.3638 | pass |

## 6. 성능 벤치마크

| Model | Batch Size | Block Size | Forward ms | Tokens/s | Train Tokens/s | Peak Memory MB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lrp_ssm_fixed_calibrated | 1 | 32 | 79.4688 | 402.6736 | 133.9058 | 270.6143 | pass |
| lrp_ssm_fixed_calibrated | 2 | 32 | 82.5282 | 775.4921 | 247.6595 | 276.7578 | pass |
| lrp_ssm_fixed_calibrated | 4 | 32 | 84.3767 | 1517.0068 | 539.9799 | 314.5718 | pass |
| lrp_ssm_fixed_calibrated | 1 | 64 | 162.1863 | 394.6079 | 137.7938 | 276.7578 | pass |
| lrp_ssm_fixed_calibrated | 2 | 64 | 161.1843 | 794.1221 | 249.8775 | 315.2910 | pass |
| lrp_ssm_fixed_calibrated | 4 | 64 | 168.9838 | 1514.9383 | 534.7781 | 416.8608 | pass |
| lrp_ssm_learned_router | 1 | 32 | 31.9449 | 1001.7236 | 167.2917 | 269.1348 | pass |
| lrp_ssm_learned_router | 2 | 32 | 32.4727 | 1970.8851 | 467.4475 | 275.2793 | pass |
| lrp_ssm_learned_router | 4 | 32 | 31.6237 | 4047.6023 | 960.1254 | 313.1729 | pass |
| lrp_ssm_learned_router | 1 | 64 | 62.4428 | 1024.9387 | 253.5269 | 275.2793 | pass |
| lrp_ssm_learned_router | 2 | 64 | 64.2239 | 1993.0269 | 480.4164 | 313.9214 | pass |
| lrp_ssm_learned_router | 4 | 64 | 63.7547 | 4015.3928 | 975.6301 | 416.5273 | pass |
| transformer | 1 | 32 | 2.0494 | 15614.3261 | 3530.5904 | 274.9863 | pass |
| transformer | 2 | 32 | 2.1788 | 29374.5066 | 6422.0404 | 281.1216 | pass |
| transformer | 4 | 32 | 1.9258 | 66465.1941 | 12144.1895 | 318.4258 | pass |
| transformer | 1 | 64 | 1.9370 | 33041.1259 | 7219.3696 | 281.1216 | pass |
| transformer | 2 | 64 | 2.1285 | 60135.6811 | 13754.9512 | 318.4609 | pass |
| transformer | 4 | 64 | 2.2736 | 112594.7819 | 24575.0249 | 421.0527 | pass |
| linear_attention | 1 | 32 | 2.0893 | 15315.9880 | 2623.3809 | 274.9614 | pass |
| linear_attention | 2 | 32 | 2.3816 | 26872.9163 | 6511.2645 | 281.0967 | pass |
| linear_attention | 4 | 32 | 2.1327 | 60017.2550 | 12548.0108 | 320.2759 | pass |
| linear_attention | 1 | 64 | 2.7927 | 22916.7263 | 6349.4205 | 281.0967 | pass |
| linear_attention | 2 | 64 | 2.8550 | 44834.2534 | 12473.2507 | 320.2759 | pass |
| linear_attention | 4 | 64 | 2.3227 | 110217.5074 | 23965.7290 | 424.7485 | pass |
| local_attention | 1 | 32 | 2.1129 | 15145.2047 | 3846.4775 | 274.9863 | pass |
| local_attention | 2 | 32 | 2.3095 | 27711.8659 | 5931.6379 | 281.1216 | pass |
| local_attention | 4 | 32 | 2.0153 | 63514.7473 | 13690.7474 | 318.4258 | pass |
| local_attention | 1 | 64 | 1.9841 | 32256.7638 | 7853.3967 | 281.1216 | pass |
| local_attention | 2 | 64 | 2.1268 | 60184.3145 | 13821.4016 | 318.4609 | pass |
| local_attention | 4 | 64 | 2.2176 | 115441.1566 | 25084.6607 | 421.0527 | pass |
| gru | 1 | 32 | 0.4427 | 72277.1831 | 6570.8959 | 272.5156 | pass |
| gru | 2 | 32 | 0.6947 | 92120.7935 | 11555.8091 | 278.6509 | pass |
| gru | 4 | 32 | 0.7695 | 166333.1341 | 28057.0083 | 315.6108 | pass |
| gru | 1 | 64 | 0.6248 | 102429.4996 | 15335.8062 | 278.6509 | pass |
| gru | 2 | 64 | 0.7404 | 172884.1946 | 28658.4274 | 315.6094 | pass |
| gru | 4 | 64 | 0.9454 | 270779.1246 | 45745.9811 | 417.7319 | pass |

## 7. Gate 분석, LRP-SSM 전용

| Metric | Value |
| --- | --- |
| mean_active_paths | 1.1534 |
| zero_gate_ratio | 0.2808 |
| all_on_gate_ratio | 0.0183 |
| calibration_current_scale | 0.0100 |
| target_active_paths | [1.0, 2.0] |

cached gate가 sparse할수록 SSM update의 path 사용량이 줄어든다. 이 비교는 gate cache를 고정한 조건이므로 online FullSNN router 비용과는 분리해서 해석해야 한다.

## 8. Ablation 권장

| Mode | Eval Loss | Eval PPL | Mean Active Paths | Zero Gate Ratio | All On Gate Ratio |
| --- | --- | --- | --- | --- | --- |
| cached_gate | 10.9010 | 54229.8703 | 0.4004 | 0.7305 | 0.0078 |
| zero_gate | 10.9010 | 54229.8703 | 0.0000 | 1.0000 | 0.0000 |
| all_on_gate | 10.9010 | 54229.7927 | 4.0000 | 0.0000 | 1.0000 |
| random_gate | 10.9010 | 54229.8186 | 1.5273 | 0.1582 | 0.0195 |
| distilled_router | 10.9010 | 54229.7669 | 2.1094 | 0.0312 | 0.0723 |

## 9. 해석

- Eval perplexity 기준 최상위 모델: transformer.
- Train throughput 기준 최상위 모델: gru.
- Peak memory 기준 최상위 모델: lrp_ssm_fixed_calibrated.
- LRP-SSM의 gate sparsity 기여는 cached/zero/all_on/random gate ablation을 함께 보아야 한다.
- Local attention은 제한된 receptive field로 memory를 줄일 수 있고, linear attention은 단순화된 prefix baseline이라 full attention과 품질 차이가 날 수 있다.
- 기본 보고서는 동일 hidden-size 조건이며 parameter count가 완전히 같지는 않다.

## 10. 한계

- C4 전체가 아니라 cache subset 기준 결과다.
- cached gate는 고정 gate이므로 end-to-end learned routing이 아니다.
- LRP-SSM과 Transformer의 parameter count가 완전히 동일하지 않을 수 있다.
- linear attention 구현은 simplified baseline이다.
- online FullSNN router runtime은 별도 비용이며 cached gate 결과와 구분해야 한다.

## 11. 결론

- 가장 좋은 eval perplexity 모델: transformer.
- 가장 빠른 모델: gru.
- memory 효율 모델: lrp_ssm_fixed_calibrated.
- LRP-SSM은 cached SNN gate 기반 recurrent state update 계열로, attention baseline과 다른 효율/품질 tradeoff를 보인다.
- 다음 개선점은 parameter-matched 설정, 더 큰 C4 cache, online router 비용 별도 측정, learned/distilled router 비교다.

Generated at: 2026-05-13T22:52:08
Raw results: `C:\Users\honka\OneDrive\Desktop\New_Neuromorphic_AI_TEST\level_test\results`
Runs: `C:\Users\honka\OneDrive\Desktop\New_Neuromorphic_AI_TEST\level_test\runs`

## LRP Fix Analysis

기존 LRP 문제는 gate sparsity, router input_dim mismatch, 작은 path contribution, 단일 layer capacity, fixed random gate였다.
Old report available: True

### Before vs After Gate Stats

| Case | Mean Active | Zero Ratio | All On Ratio | Router Input Dim | Model Dim | Current Scale |
| --- | --- | --- | --- | --- | --- | --- |
| before_lrp_ssm_old | 0.3765 | 0.7023 | 0.0028 | 64 | 64 | 0.0100 |
| after_fixed_calibrated | 1.1534 | 0.2808 | 0.0183 | 64 | 64 | 1.0000 |
| after_learned_router | 2.0000 | 0.0000 | 0.0000 | learned | 64 | n/a |

### Before vs After Path Contribution

| Mode | Gamma | Raw Delta | Scaled Delta | Path/Base | Logits L2 vs Zero |
| --- | --- | --- | --- | --- | --- |
| zero_gate | 0.1013 | 0.0000 | 0.0086 | 0.0013 | 0.0000 |
| cached_gate | 0.1013 | 0.0143 | 0.3111 | 0.0442 | 0.0711 |
| all_on_gate | 0.1013 | 0.0416 | 0.6551 | 0.0908 | 0.1640 |
| random_gate_same_density | 0.1013 | 0.0141 | 0.3121 | 0.0442 | 0.0701 |
| force_min_active_1 | 0.1013 | 0.0233 | 0.4740 | 0.0666 | 0.1055 |
| force_min_active_2 | 0.1013 | 0.0344 | 0.5993 | 0.0834 | 0.1445 |

### Before vs After LM Quality

| Case | Train Loss | Eval Loss | Eval PPL | Uniform Loss Gap |
| --- | --- | --- | --- | --- |
| before_lrp_ssm_old | 10.0699 | 10.2003 | 26910.8850 | -0.6246 |
| after_fixed_calibrated | 10.1392 | 10.2434 | 28096.9531 | -0.5815 |
| after_learned_router | 10.0699 | 10.2003 | 26910.8850 | -0.6246 |

### Updated Model Comparison

| Model | Params | Eval Loss | Eval PPL | Uniform Gap | Status |
| --- | --- | --- | --- | --- | --- |
| lrp_ssm_fixed_calibrated | 6475843 | 10.2434 | 28096.9531 | -0.5815 | pass |
| lrp_ssm_learned_router | 6484679 | 10.2003 | 26910.8850 | -0.6246 | pass |
| transformer | 6587072 | 10.0036 | 22105.4477 | -0.8213 | pass |
| linear_attention | 6586496 | 10.0548 | 23267.6413 | -0.7701 | pass |
| local_attention | 6587072 | 10.0126 | 22305.1893 | -0.8123 | pass |
| gru | 6507776 | 10.0714 | 23657.2020 | -0.7535 | pass |

### Throughput/Memory

| Model | Train Tokens/s | Eval Tokens/s | Forward ms | Peak GPU Memory MB |
| --- | --- | --- | --- | --- |
| lrp_ssm_fixed_calibrated | 464.9307 | 787.4400 | 79.4688 | 364.7549 |
| lrp_ssm_learned_router | 548.6990 | 1231.8175 | 31.9449 | 365.9155 |
| transformer | 22256.8374 | 58928.2067 | 2.0494 | 366.5283 |
| linear_attention | 19797.3822 | 57443.1318 | 2.0893 | 366.5166 |
| local_attention | 22285.7224 | 61847.7959 | 2.1129 | 366.5283 |
| gru | 40123.2405 | 129289.6518 | 0.4427 | 365.3638 |
