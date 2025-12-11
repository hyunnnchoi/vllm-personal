# vLLM HOL Blocking 추적 가이드

## 개요

vLLM 스케줄러에 HOL (Head-of-Line) Blocking을 추적하는 디버깅 로직이 추가되었습니다. 이를 통해 FCFS 스케줄링 정책에서 발생하는 HOL blocking을 상세히 분석할 수 있습니다.

## HOL Blocking이란?

HOL Blocking은 큐의 맨 앞에 있는 요청이 리소스 부족(예: KV cache 공간 부족)으로 처리되지 못할 때, 뒤에 있는 다른 요청들도 함께 대기하게 되는 현상입니다. FCFS 스케줄링에서는 순서를 지키기 때문에 뒤에 있는 작은 요청들이 리소스가 충분함에도 불구하고 처리되지 못합니다.

## 사용 방법

### 1. HOL Blocking 추적 활성화

환경 변수를 설정하여 HOL blocking 추적을 활성화합니다:

```bash
export VLLM_HOL_TRACKING=1
export VLLM_HOL_LOG_DIR="./hol_blocking_logs"  # 선택사항, 기본값: ./hol_blocking_logs
```

### 2. vLLM 벤치마크 실행

```bash
cd /home/xsailor6/hmchoi/vllm-deployments/vanilla-vllm

# HOL tracking 활성화하여 벤치마크 실행
VLLM_HOL_TRACKING=1 VLLM_HOL_LOG_DIR="./hol_logs" bash vllm-bench-serve.sh
```

또는 직접 vLLM 서버를 실행할 때:

```bash
VLLM_HOL_TRACKING=1 python -m vllm.entrypoints.openai.api_server \
    --model <model_name> \
    --scheduling-policy fcfs \
    ...
```

### 3. 로그 분석

벤치마크가 완료되면 생성된 CSV 파일을 분석합니다:

```bash
python /home/xsailor6/hmchoi/analyze_hol_blocking.py ./hol_blocking_logs
```

또는 특정 CSV 파일을 지정:

```bash
python /home/xsailor6/hmchoi/analyze_hol_blocking.py ./hol_blocking_logs/hol_blocking_20251211_123456.csv
```

## CSV 로그 파일 구조

생성되는 CSV 파일에는 다음 컬럼들이 포함됩니다:

| 컬럼명 | 설명 |
|--------|------|
| `scheduling_step` | 스케줄링 단계 번호 |
| `request_id` | 요청 ID |
| `arrival_time` | 요청 도착 시간 (Unix timestamp) |
| `queue_enter_time` | 큐 진입 시간 |
| `scheduled_time` | 스케줄된 시간 |
| `finish_time` | 완료 시간 |
| `queue_position` | 큐에서의 위치 (0부터 시작) |
| `num_waiting_requests` | 대기 중인 요청 수 |
| `event_type` | 이벤트 타입: `queued`, `blocking`, `scheduled`, `finished` |
| `blocking_reason` | 블로킹 이유 (아래 참조) |
| `token_budget_remaining` | 남은 토큰 예산 |
| `num_running_requests` | 실행 중인 요청 수 |
| `wait_duration_ms` | 대기 시간 (밀리초) |

### 블로킹 이유 (`blocking_reason`)

- `kv_cache_full`: KV cache 공간 부족 (직접 블로킹)
- `blocked_by_hol`: 앞의 요청이 블로킹되어 간접적으로 블로킹됨 (HOL blocking)
- `max_running_reqs`: 최대 실행 요청 수 도달
- `token_budget_insufficient`: 토큰 예산 부족
- `encoder_budget_insufficient`: Encoder 예산 부족

## 분석 예시

분석 스크립트는 다음과 같은 정보를 제공합니다:

```
=== HOL Blocking 분석 ===
직접 블로킹 (KV cache full): 150
간접 블로킹 (HOL에 의한): 450
평균 간접/직접 블로킹 비율: 3.00

=== 대기 시간 통계 (ms) ===
평균: 1234.56
중앙값: 890.12
최소: 10.34
최대: 5678.90
```

이 결과는 HOL blocking으로 인해 평균적으로 직접 블로킹된 요청 1개당 3개의 요청이 추가로 블로킹되었음을 보여줍니다.

## 추가 분석

더 상세한 분석을 위해 Python으로 직접 CSV를 읽어 분석할 수 있습니다:

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSV 읽기
df = pd.read_csv('./hol_blocking_logs/hol_blocking_20251211_123456.csv')

# 시간에 따른 블로킹 이벤트 수 시각화
blocking_df = df[df['event_type'] == 'blocking']
blocking_per_step = blocking_df.groupby('scheduling_step').size()
plt.plot(blocking_per_step.index, blocking_per_step.values)
plt.xlabel('Scheduling Step')
plt.ylabel('Number of Blocking Events')
plt.title('HOL Blocking Events Over Time')
plt.show()

# 블로킹 이유별 분포
blocking_reasons = blocking_df['blocking_reason'].value_counts()
blocking_reasons.plot(kind='bar')
plt.xlabel('Blocking Reason')
plt.ylabel('Count')
plt.title('Distribution of Blocking Reasons')
plt.show()
```

## 성능 영향

HOL tracking은 각 스케줄링 단계에서 추가적인 로깅을 수행하므로 약간의 성능 오버헤드가 있을 수 있습니다. 프로덕션 환경에서는 비활성화하고, 성능 분석 시에만 활성화하는 것을 권장합니다.

## 문제 해결

### 로그 파일이 생성되지 않는 경우

1. 환경 변수가 올바르게 설정되었는지 확인:
   ```bash
   echo $VLLM_HOL_TRACKING
   ```

2. 로그 디렉토리에 쓰기 권한이 있는지 확인:
   ```bash
   ls -ld $VLLM_HOL_LOG_DIR
   ```

3. vLLM 로그에서 HOL tracker 초기화 메시지 확인:
   ```
   INFO: HOL Blocking tracker initialized. Log file: ...
   ```

### CSV 파일이 너무 큰 경우

HOL tracking은 매 스케줄링 단계마다 이벤트를 기록하므로, 긴 벤치마크의 경우 파일이 매우 커질 수 있습니다. 필요한 경우 벤치마크 시간을 줄이거나, 로그를 주기적으로 분석하고 삭제하세요.

## 참고

- 이 기능은 주로 FCFS 스케줄링 정책에서 HOL blocking을 분석하기 위해 설계되었습니다.
- 다른 스케줄링 정책(예: priority, ISRTF)에서도 사용할 수 있지만, HOL blocking의 의미가 다를 수 있습니다.
- 코드 수정 사항은 `/home/xsailor6/hmchoi/vllm/vllm/v1/core/sched/scheduler.py`에 있으며, `[NOTE, hyunnnchoi, 2025.12.11]` 주석으로 표시되어 있습니다.

