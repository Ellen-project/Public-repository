# Low Rank Path based SSM vs Attention Baselines

## 1. 실험 목적

- Low Rank Path based SSM과 attention 계열 baseline의 language modeling 성능 및 효율 비교.
- C4 cache 기반 동일 데이터 조건에서 비교.

## 2. 실험 환경

| 항목 | 값 |
| --- | --- |
| Python | 3.12.9 |
| PyTorch | 2.12.0+cu126 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| device | cuda |
| AMP | False |
| seed | 1 |
| token cache | ../runs/cache/c4_token_cache_strong_router.pt |
| gate cache | ../runs/cache/c4_gate_cache_strong_router.pt |

## 3. 비교 모델

| Model | Type | Params | Uses Gate | Complexity | Notes |
| --- | --- | --- | --- | --- | --- |
| lrp_ssm | Low-rank path SSM | n/a | yes | O(T*paths*rank) | cached/learned gates |
| lrp_ssm_strong_path_bias_decay | Low-rank path SSM strong routed path-bias decay | n/a | yes | O(T*paths*rank) | cached/learned gates |
| transformer | Full causal attention | 13675520 | no | O(T^2) |  |
| linear_attention | Kernelized linear attention | 13673984 | no | O(T), simplified ELU+1 prefix | simplified baseline |
| local_attention | Sliding-window attention | 13675520 | no | O(T*w), w=64 |  |
| gru | Recurrent | 13262080 | no | O(T) recurrent |  |

## 4. 학습 설정

| 항목 | 값 |
| --- | --- |
| block_size | n/a |
| batch_size | 8 |
| max_train_steps | 1000 |
| optimizer | AdamW |
| lr | 0.0003 |
| weight_decay | 0.0100 |
| grad_clip | 1.0000 |
| model_dim | 128 |
| num_layers | 4 |
| num_heads | 4 |
| state_dim | 128 |
| rank | 4 |
| num_paths | 8 |

## 5. 결과 요약

| Model | Params | Best Eval Loss | Best Eval PPL | Final Train Loss | Avg Train Tokens/s | Avg Eval Tokens/s | Peak GPU Memory MB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lrp_ssm | n/a | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | failed |
| lrp_ssm_strong_path_bias_decay | n/a | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | failed |
| transformer | 13675520 | 7.0110 | 1108.7137 | 6.0941 | 38959.8118 | 121368.8484 | 1010.7222 | pass |
| linear_attention | 13673984 | 7.0029 | 1099.8377 | 6.1526 | 36842.5954 | 134138.2029 | 1065.8989 | pass |
| local_attention | 13675520 | 7.0178 | 1116.3607 | 6.0905 | 42255.7356 | 137812.8336 | 1010.7222 | pass |
| gru | 13262080 | 7.6493 | 2099.0794 | 7.5345 | 52314.9478 | 180015.0806 | 1005.1001 | pass |

## 6. 성능 벤치마크

| Model | Batch Size | Block Size | Forward ms | Tokens/s | Train Tokens/s | Peak Memory MB | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| not available | n/a | n/a | n/a | n/a | n/a | n/a | not available |

## 7. Gate 분석, LRP-SSM 전용

| Metric | Value |
| --- | --- |
| mean_active_paths | n/a |
| zero_gate_ratio | n/a |
| all_on_gate_ratio | n/a |
| calibration_current_scale | n/a |
| target_active_paths | n/a |

cached gate가 sparse할수록 SSM update의 path 사용량이 줄어든다. 이 비교는 gate cache를 고정한 조건이므로 online FullSNN router 비용과는 분리해서 해석해야 한다.

## 8. Ablation 권장

Ablation reports are not available.

## 9. 해석

- Eval perplexity 기준 최상위 모델: linear_attention.
- Train throughput 기준 최상위 모델: gru.
- Peak memory 기준 최상위 모델: lrp_ssm.
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

- 가장 좋은 eval perplexity 모델: linear_attention.
- 가장 빠른 모델: gru.
- memory 효율 모델: lrp_ssm.
- LRP-SSM은 cached SNN gate 기반 recurrent state update 계열로, attention baseline과 다른 효율/품질 tradeoff를 보인다.
- 다음 개선점은 parameter-matched 설정, 더 큰 C4 cache, online router 비용 별도 측정, learned/distilled router 비교다.

Generated at: 2026-05-24T17:06:23
Raw results: `C:\Dev\2025\neuromorphicAI\experiments\results`
Runs: `C:\Dev\2025\neuromorphicAI\experiments\runs`

## LRP Fix Analysis

기존 LRP 문제는 gate sparsity, router input_dim mismatch, 작은 path contribution, 단일 layer capacity, fixed random gate였다.
Old report available: False

### Before vs After Gate Stats

| Case | Mean Active | Zero Ratio | All On Ratio | Router Input Dim | Model Dim | Current Scale |
| --- | --- | --- | --- | --- | --- | --- |
| before_lrp_ssm_old | n/a | n/a | n/a | n/a | n/a | n/a |
| after_fixed_calibrated | 1.3566 | 0.2369 | 0.0002 | 128 | 128 | 0.0018 |
| after_learned_router | n/a | n/a | n/a | learned | n/a | n/a |
| after_hybrid | n/a | n/a | n/a | cached+learned | n/a | n/a |

### Before vs After Path Contribution

| Mode | Gamma | Raw Delta | Scaled Delta | Path/Base | Logits L2 vs Zero |
| --- | --- | --- | --- | --- | --- |
| not available | n/a | n/a | n/a | n/a | n/a |

### Before vs After LM Quality

| Case | Train Loss | Eval Loss | Eval PPL | Uniform Loss Gap |
| --- | --- | --- | --- | --- |
| before_lrp_ssm_old | n/a | n/a | n/a | n/a |
| after_fixed_calibrated | n/a | n/a | n/a | n/a |
| after_learned_router | n/a | n/a | n/a | n/a |

### Updated Model Comparison

| Model | Params | Eval Loss | Eval PPL | Uniform Gap | Status |
| --- | --- | --- | --- | --- | --- |
| lrp_ssm | n/a | n/a | n/a | n/a | failed |
| lrp_ssm_strong_path_bias_decay | n/a | n/a | n/a | n/a | failed |
| transformer | 13675520 | 7.0110 | 1108.7137 | -3.8139 | pass |
| linear_attention | 13673984 | 7.0029 | 1099.8377 | -3.8220 | pass |
| local_attention | 13675520 | 7.0178 | 1116.3607 | -3.8071 | pass |
| gru | 13262080 | 7.6493 | 2099.0794 | -3.1757 | pass |

### Throughput/Memory

| Model | Train Tokens/s | Eval Tokens/s | Forward ms | Peak GPU Memory MB |
| --- | --- | --- | --- | --- |
| lrp_ssm | 0.0000 | 0.0000 | n/a | 0.0000 |
| lrp_ssm_strong_path_bias_decay | 0.0000 | 0.0000 | n/a | 0.0000 |
| transformer | 38959.8118 | 121368.8484 | n/a | 1010.7222 |
| linear_attention | 36842.5954 | 134138.2029 | n/a | 1065.8989 |
| local_attention | 42255.7356 | 137812.8336 | n/a | 1010.7222 |
| gru | 52314.9478 | 180015.0806 | n/a | 1005.1001 |
