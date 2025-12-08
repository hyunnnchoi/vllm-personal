# ISRTF 메트릭 분석 가이드

## 📊 개요

ISRTF 스케줄러의 성능을 평가하기 위한 상세 메트릭 시스템입니다.

## 🎯 측정 메트릭

### 1. 예측 정확도 (Prediction Accuracy)

- **Initial Prediction**: 요청 도착 시 초기 예측값
- **Final Prediction**: 마지막 업데이트된 예측값
- **Actual Output Tokens**: 실제 생성된 토큰 수
- **Prediction Error**: `|predicted - actual|` 절대 오차
- **Prediction Error Rate**: `error / actual` 상대 오차율

### 2. Kendall's Tau 상관계수

**의미**: 예측 순서와 실제 순서 간의 일치도 (-1 ~ 1)

- **τ = +1**: 완벽히 동일한 순서로 예측 (perfect prediction!)
- **τ = -1**: 정확히 반대 순서로 예측 (worst prediction!)
- **τ = 0**: 순서에 상관관계 없음 (random prediction)

**예시**:
```
Request A: predicted=50,  actual=50  → τ=1 (완벽)
Request B: predicted=100, actual=100
Request C: predicted=150, actual=150

예측 순서: A < B < C
실제 순서: A < B < C  ✓ 일치!
```

```
Request A: predicted=50,  actual=150 → τ=-1 (최악)
Request B: predicted=100, actual=100
Request C: predicted=150, actual=50

예측 순서: A < B < C
실제 순서: C < B < A  ✗ 정반대!
```

**해석 기준**:
- **τ > 0.7**: 강한 양의 상관관계 (예측이 매우 정확)
- **τ > 0.4**: 중간 정도의 양의 상관관계
- **τ > 0.2**: 약한 양의 상관관계
- **τ ≈ 0**: 상관관계 없음 (예측 무의미)
- **τ < 0**: 음의 상관관계 (예측이 오히려 역효과)

**계산 방법**: 
- 100개의 완료된 요청마다 계산
- 예측된 remaining tokens 순서 vs 실제 output tokens 순서 비교
- p-value도 함께 계산하여 통계적 유의성 검증

### 3. 레이턴시 메트릭

- **Total Latency**: 요청 도착 ~ 완료까지 총 시간
- **Queue Wait Time**: 큐에서 대기한 시간
- **Time to First Token (TTFT)**: 첫 토큰까지 걸린 시간

## 🚀 사용 방법

### Step 1: vLLM 서버 실행 (메트릭 활성화)

```bash
# 환경 변수로 메트릭 설정
export VLLM_ISRTF_METRICS_DIR="/home/xsailor6/hmchoi/isrtf_metrics"
export VLLM_ISRTF_DETAILED_LOGGING="true"
export VLLM_ISRTF_KENDALL_WINDOW="100"

# ISRTF 서버 시작
cd /home/xsailor6/hmchoi/vllm-deployments/vanilla-vllm
./vanilla_vllm_isrtf.sh
```

### Step 2: 요청 전송

```bash
cd /home/xsailor6/hmchoi/vllm/kukt/request_generator

python send_to_vllm.py \
  --model "meta-llama/Llama-3.1-8B" \
  --input /data/test_prompts.json \
  --batch-size 10
```

### Step 3: 메트릭 분석

```bash
cd /home/xsailor6/hmchoi/vllm

# 분석 스크립트 실행
python tools/analyze_isrtf_metrics.py \
  --log-dir /home/xsailor6/hmchoi/isrtf_metrics \
  --output-dir /home/xsailor6/hmchoi/isrtf_analysis
```

## 📁 출력 파일

### CSV 파일 (자동 생성)

```
isrtf_metrics/
├── request_metrics.csv          # 요청별 상세 메트릭
├── kendall_tau.csv               # Kendall's tau 시계열
├── prediction_history.csv        # 예측 업데이트 이력
└── summary.json                  # 요약 통계
```

### 분석 결과 (analyze_isrtf_metrics.py 실행 후)

```
isrtf_analysis/
├── prediction_accuracy.png       # 예측 정확도 그래프
├── kendall_tau_analysis.png      # Kendall's tau 분석
└── latency_analysis.png          # 레이턴시 분석
```

## 📊 CSV 파일 상세

### request_metrics.csv

| 컬럼 | 설명 |
|------|------|
| request_id | 요청 고유 ID |
| arrival_time | 요청 도착 시간 |
| completion_time | 요청 완료 시간 |
| total_latency | 총 레이턴시 (초) |
| queue_wait_time | 큐 대기 시간 (초) |
| time_to_first_token | 첫 토큰까지 시간 (초) |
| initial_prediction | 초기 예측값 |
| final_prediction | 최종 예측값 |
| actual_output_tokens | 실제 출력 토큰 수 |
| prediction_error | 예측 오차 (절대값) |
| prediction_error_rate | 예측 오차율 (상대값) |
| num_prediction_updates | 예측 업데이트 횟수 |

### kendall_tau.csv

| 컬럼 | 설명 |
|------|------|
| timestamp | 측정 시각 |
| kendall_tau | Kendall's tau 값 |
| p_value | 통계적 유의성 |
| sample_size | 샘플 크기 |
| mean_prediction_error | 평균 예측 오차 |
| median_prediction_error | 중간값 예측 오차 |

### prediction_history.csv

| 컬럼 | 설명 |
|------|------|
| request_id | 요청 ID |
| num_output_tokens | 현재 생성된 토큰 수 |
| predicted_remaining | 예측된 남은 토큰 수 |
| timestamp | 예측 시각 |

## 🔧 환경 변수

```bash
# 메트릭 로그 디렉토리
export VLLM_ISRTF_METRICS_DIR="/tmp/isrtf_metrics"

# 상세 로깅 활성화 (prediction history 저장)
export VLLM_ISRTF_DETAILED_LOGGING="true"

# Kendall's tau 계산 윈도우 크기
export VLLM_ISRTF_KENDALL_WINDOW="100"
```

## 📈 분석 예시

### Python에서 직접 분석

```python
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

# Load metrics
df = pd.read_csv('/tmp/isrtf_metrics/request_metrics.csv')

# Calculate overall Kendall's tau
predictions = df['final_prediction'].values
actuals = df['actual_output_tokens'].values
tau, p_value = kendalltau(predictions, actuals)

print(f"Kendall's tau: {tau:.4f} (p={p_value:.4f})")

# Prediction accuracy
mean_error = df['prediction_error'].mean()
median_error = df['prediction_error'].median()
mean_error_rate = df['prediction_error_rate'].mean() * 100

print(f"Mean prediction error: {mean_error:.1f} tokens")
print(f"Median prediction error: {median_error:.1f} tokens")
print(f"Mean error rate: {mean_error_rate:.1f}%")

# Latency statistics
print(f"\nLatency metrics:")
print(f"  Mean: {df['total_latency'].mean():.2f}s")
print(f"  P50: {df['total_latency'].quantile(0.5):.2f}s")
print(f"  P95: {df['total_latency'].quantile(0.95):.2f}s")
print(f"  P99: {df['total_latency'].quantile(0.99):.2f}s")
```

### Pandas로 간단 분석

```bash
# CSV 파일 미리보기
head -20 /tmp/isrtf_metrics/request_metrics.csv

# 요약 통계
python -c "
import pandas as pd
df = pd.read_csv('/tmp/isrtf_metrics/request_metrics.csv')
print(df.describe())
"
```

## 🎨 시각화 예시

`analyze_isrtf_metrics.py`가 생성하는 그래프:

### 1. Prediction Accuracy
- Predicted vs Actual scatter plot (Kendall's tau 표시)
- Prediction error distribution
- Error rate distribution  
- Error vs output length

### 2. Kendall's Tau Analysis
- Kendall's tau over time (윈도우별)
- Prediction error over time

### 3. Latency Analysis
- Total latency distribution
- Queue wait time distribution
- TTFT distribution
- Latency vs output length

## 💡 해석 가이드

### 좋은 결과 (ISRTF가 잘 작동)

✓ Kendall's tau > 0.6  
✓ Mean error rate < 30%  
✓ Median error < 50 tokens  
✓ P95 latency가 FCFS 대비 낮음

### 개선이 필요한 경우

⚠️ Kendall's tau < 0.3 → 예측 모델 재학습 필요  
⚠️ Mean error rate > 50% → 더 많은 학습 데이터 필요  
⚠️ Error가 시간에 따라 증가 → 드리프트 문제

## 🔍 디버깅

### 메트릭이 생성되지 않는 경우

```bash
# 1. 로그 확인
sudo docker logs vllm 2>&1 | grep "ISRTF Metrics"

# 2. 디렉토리 확인
ls -la /tmp/isrtf_metrics/

# 3. 컨테이너 내부 확인
sudo docker exec -it vllm ls -la /tmp/isrtf_metrics/
```

### 메트릭 파일이 비어있는 경우

- 요청이 완료되지 않았을 수 있음
- ISRTF 정책이 활성화되지 않았을 수 있음
- 예측 모델이 로드되지 않았을 수 있음

## 📝 FCFS vs ISRTF 비교

### 비교 테스트 실행

```bash
# 1. FCFS 테스트
export VLLM_ISRTF_METRICS_DIR="/tmp/fcfs_metrics"
# vanilla_vllm.sh 실행 (FCFS 모드)
# 요청 전송

# 2. ISRTF 테스트
export VLLM_ISRTF_METRICS_DIR="/tmp/isrtf_metrics"
# vanilla_vllm_isrtf.sh 실행 (ISRTF 모드)
# 동일한 요청 전송

# 3. 비교 분석
python compare_schedulers.py \
  --fcfs-dir /tmp/fcfs_metrics \
  --isrtf-dir /tmp/isrtf_metrics
```

### 비교 지표

- **Average Latency Reduction**: ISRTF가 얼마나 latency를 줄였는지
- **P95/P99 Improvement**: tail latency 개선
- **Throughput**: 처리량 차이
- **Scheduling Quality**: Kendall's tau (ISRTF만 해당)

## 🎯 다음 단계

1. **기본 메트릭 확인**: Kendall's tau > 0.5인지 확인
2. **예측 정확도 분석**: error rate가 합리적인지
3. **FCFS 비교**: 실제 latency 개선 효과 측정
4. **모델 개선**: 필요시 더 많은 데이터로 재학습
5. **하이퍼파라미터 튜닝**: prediction interval 조정

---

**작성일**: 2025.12.07  
**작성자**: hyunnnchoi

