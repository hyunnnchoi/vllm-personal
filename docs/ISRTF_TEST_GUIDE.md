# ISRTF with ELIS 예측 모델 테스트 가이드

## 📋 개요

ISRTF (Iterative Shortest Remaining Time First) 스케줄링 정책을 ELIS 예측 모델과 함께 vLLM에서 테스트하는 가이드입니다.

## 🏗️ 현재 구현 상태

### 1. 주요 변경사항 (최근 커밋 기준)

**커밋 히스토리:**
- `262fbddda` - Update scheduler and request queue implementation with custom request generator
- `9b8176da9` - Fix tokenizer loading to use model_name instead of hardcoded path
- `5ded9e9aa` - feat: request sender for reimplementing ELIS

### 2. 구현된 컴포넌트

#### A. 스케줄러 설정 (`vllm/config/scheduler.py`)
```python
# Line 21-23: ISRTF 정책 추가
SchedulerPolicy = Literal["fcfs", "priority", "isrtf"]

# Line 109-117: 정책 설명
policy: SchedulerPolicy = "fcfs"
- "fcfs": First Come First Served
- "priority": 우선순위 기반
- "isrtf": ELIS 예측 모델 기반 Shortest Remaining Time First
```

#### B. Request Queue (`vllm/v1/core/sched/request_queue.py`)
- **ISRTFRequestQueue** 클래스 구현 (Line 222+)
  - `predicted_remaining_tokens` 기반 우선순위 큐
  - Min-heap 자료구조 사용
  - 50토큰마다 예측 업데이트 시 우선순위 재조정

#### C. Scheduler (`vllm/v1/core/sched/scheduler.py`)
- **ELIS 예측 모델 초기화** (Line 308+)
  - BGE 모델 (BAAI/bge-base-en-v1.5) + 8 FC layers
  - Checkpoint 로딩
- **예측 메서드**:
  - `_init_elis_predictor()`: 모델 초기화
  - `_predict_remaining_tokens()`: 텍스트 입력 → 예측
  - `_update_elis_prediction()`: 50토큰마다 예측 업데이트
  - `_make_initial_prediction()`: 신규 요청 초기 예측

#### D. Request 클래스 (`vllm/v1/request.py`)
새로운 필드 추가 (Line 132-134):
```python
self.predicted_remaining_tokens: float = float('inf')
self.last_prediction_at_tokens: int = 0
self.prediction_history: list[tuple[int, float]] = []
```

### 3. 예측 모델

**위치:** `/home/xsailor6/hmchoi/ELIS/train/checkpoints/latest_model.pt` (499MB)

**아키텍처:**
- BGE 임베딩 (frozen, 768차원)
- Mean pooling
- 8개의 FC layers (1024 hidden dim)
- 출력: 스칼라 (예측 remaining tokens)

## 🚀 테스트 방법

### 방법 1: 환경 변수 설정으로 ISRTF 활성화

#### Step 1: vLLM 컨테이너 실행 스크립트 수정

`vllm-deployments/vanilla-vllm/vanilla_vllm.sh` 파일을 수정하여 ISRTF 환경 변수를 추가:

```bash
#!/bin/bash
# [NOTE, hyunnnchoi, 2025.12.07] Enable ISRTF scheduling with ELIS predictor

IMAGE_NAME="${IMAGE_NAME:-potato4332/vanilla-vllm:v0.11.0-debug}"
MODEL_NAME="${MODEL_NAME:-upstage/SOLAR-10.7B-v1.0}"
HF_TOKEN="${HF_TOKEN:-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}"

# Stop and remove existing container
sudo docker stop vllm 2>/dev/null || true
sudo docker rm vllm 2>/dev/null || true

# Create directories
mkdir -p vllm_logs
mkdir -p nsys_reports

# Start vLLM server with ISRTF scheduling
sudo docker run -d --name vllm --gpus all --ipc=host \
  -p 8000:8000 \
  -v /home/xsailor6/hmchoi/ELIS:/ELIS \
  -v /home/xsailor6/hmchoi/ELIS/data:/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -e VLLM_SCHEDULER_POLICY=isrtf \
  -e VLLM_ELIS_CHECKPOINT=/ELIS/train/checkpoints/latest_model.pt \
  -e VLLM_ELIS_BGE_MODEL=BAAI/bge-base-en-v1.5 \
  -e VLLM_ELIS_PREDICTION_INTERVAL=50 \
  -e VLLM_ELIS_PATH=/ELIS \
  ${IMAGE_NAME} \
  vllm serve ${MODEL_NAME} \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --scheduling-policy isrtf

# Follow logs
sudo docker logs -f vllm
```

**주요 환경 변수:**
- `VLLM_SCHEDULER_POLICY=isrtf`: ISRTF 스케줄링 활성화
- `VLLM_ELIS_CHECKPOINT`: 예측 모델 체크포인트 경로
- `VLLM_ELIS_BGE_MODEL`: BGE 모델 이름
- `VLLM_ELIS_PREDICTION_INTERVAL`: 예측 업데이트 간격 (기본 50 토큰)
- `VLLM_ELIS_PATH`: ELIS 모듈 경로

#### Step 2: 컨테이너 실행

```bash
cd /home/xsailor6/hmchoi/vllm-deployments/vanilla-vllm
bash vanilla_vllm.sh
```

#### Step 3: 로그 확인

예측 모델이 정상적으로 로드되었는지 확인:

```bash
sudo docker logs vllm 2>&1 | grep -i "ELIS"
```

기대되는 로그:
```
[ELIS] Initializing predictor...
[ELIS] Checkpoint: /ELIS/train/checkpoints/latest_model.pt
[ELIS] BGE model: BAAI/bge-base-en-v1.5
[ELIS] Prediction interval: 50 tokens
[ELIS] Predictor initialized successfully on cuda
[ELIS] Model epoch: ...
```

### 방법 2: 명령줄 인자로 ISRTF 활성화

vLLM 서버 실행 시 `--policy` 플래그 사용:

```bash
vllm serve <MODEL_NAME> \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --scheduling-policy isrtf
```

### 방법 3: 요청 전송 및 테스트

#### A. 테스트 프롬프트 준비

```bash
# 테스트용 프롬프트 JSON 생성
cat > /home/xsailor6/hmchoi/ELIS/data/test_prompts.json << 'EOF'
{
  "prompts": [
    "Write a short story about a robot learning to paint.",
    "Explain quantum computing in simple terms.",
    "Describe the process of photosynthesis step by step.",
    "What are the key differences between Python and JavaScript?",
    "Write a poem about the ocean."
  ]
}
EOF
```

#### B. Request Generator 실행

```bash
cd /home/xsailor6/hmchoi/vllm/kukt/request_generator

python send_to_vllm.py \
  --input /home/xsailor6/hmchoi/ELIS/data/test_prompts.json \
  --output /home/xsailor6/hmchoi/ELIS/data/test_results.json \
  --server-url http://localhost:8000/v1/completions \
  --model upstage/SOLAR-10.7B-v1.0 \
  --batch-size 5 \
  --temperature 0.7
```

#### C. 실시간 로그 모니터링

다른 터미널에서:

```bash
sudo docker logs -f vllm 2>&1 | grep -E "ELIS|ISRTF|prediction"
```

기대되는 로그 패턴:
```
[ELIS] Request abc123... initial prediction: 150.3 tokens
[ELIS] Request abc123... prediction updated: 150.3 -> 98.5 (at 50 tokens)
[ELIS] Request xyz789... initial prediction: 75.2 tokens
```

## 📊 테스트 시나리오

### 시나리오 1: FCFS vs ISRTF 비교

1. **FCFS 모드로 실행** (baseline)
   ```bash
   # vanilla_vllm.sh에서 --policy isrtf 제거
   bash vanilla_vllm.sh
   # 요청 전송 및 latency 측정
   ```

2. **ISRTF 모드로 실행**
   ```bash
   # --policy isrtf 추가
   bash vanilla_vllm.sh
   # 동일한 요청 전송 및 latency 측정
   ```

3. **비교 지표**:
   - Average Time to First Token (TTFT)
   - Average Time Between Tokens (TBT)
   - P50, P95, P99 latency
   - Throughput (requests/sec)

### 시나리오 2: 다양한 길이의 요청 혼합

```python
# 짧은 요청 (예상 output: 50-100 tokens)
# 중간 요청 (예상 output: 100-300 tokens)
# 긴 요청 (예상 output: 300-500 tokens)
```

**기대 결과:**
- ISRTF가 짧은 요청을 우선 처리하여 평균 latency 감소
- 예측이 정확할수록 성능 향상

### 시나리오 3: 예측 업데이트 관찰

```bash
# 긴 output을 생성하는 요청 전송
# 50토큰마다 예측이 업데이트되는지 로그 확인
sudo docker logs -f vllm 2>&1 | grep "prediction updated"
```

## 🔍 디버깅 팁

### 1. 예측 모델이 로드되지 않는 경우

**확인 사항:**
- 체크포인트 파일 경로가 올바른지
- ELIS 디렉토리가 컨테이너에 마운트되었는지
- BGE 모델이 다운로드 가능한지 (인터넷 연결)

```bash
# 컨테이너 내부 파일 확인
sudo docker exec -it vllm ls -la /ELIS/train/checkpoints/
```

### 2. ISRTF 정책이 작동하지 않는 경우

**확인 사항:**
- vLLM 버전에 ISRTF 코드가 포함되었는지
- `--policy isrtf` 플래그가 올바르게 전달되었는지
- 환경 변수가 제대로 설정되었는지

```bash
# vLLM 설정 확인
sudo docker exec -it vllm python -c "from vllm.config import SchedulerConfig; print(SchedulerConfig.__annotations__['policy'])"
```

### 3. 예측 정확도 확인

**로그에서 prediction_history 추출:**
```bash
sudo docker logs vllm 2>&1 | grep "prediction_history" > /tmp/predictions.log
```

**분석:**
- 예측값이 실제 remaining tokens와 얼마나 가까운지
- 50토큰마다 예측이 개선되는지

## 📈 성능 측정

### 메트릭 수집

vLLM은 다음 메트릭을 자동으로 로깅합니다:
- Request latency
- Token generation time
- Queue wait time
- Scheduling decisions

```bash
# CSV 로그 파일 확인
sudo docker exec -it vllm ls -la /tmp/vllm_logs/
```

### Prometheus 메트릭

vLLM 서버는 Prometheus 엔드포인트를 제공합니다:
```bash
curl http://localhost:8000/metrics
```

## 🎯 다음 단계

1. **A/B 테스트**: FCFS vs ISRTF 성능 비교
2. **예측 모델 파인튜닝**: 현재 워크로드에 맞게 재학습
3. **하이퍼파라미터 튜닝**: 
   - `VLLM_ELIS_PREDICTION_INTERVAL` 조정 (50 → 25 or 100)
   - GPU memory utilization 최적화
4. **멀티 모델 테스트**: 다양한 모델에서 ISRTF 효과 검증

## 📝 주의사항

1. **예측 모델 오버헤드**: 
   - BGE 모델 추론 시간이 스케줄링에 추가됨
   - 배치 크기가 작을 때는 오버헤드가 더 눈에 띌 수 있음

2. **메모리 사용량**:
   - ELIS 예측 모델이 추가 GPU 메모리 사용 (약 1-2GB)
   - `--gpu-memory-utilization` 값을 적절히 조정

3. **토크나이저 일관성**:
   - 예측 모델 학습 시 사용한 토크나이저와 vLLM 토크나이저가 동일해야 함
   - 현재 vLLM은 모델별 토크나이저를 자동으로 로드

## 🤔 FAQ

**Q: ISRTF가 항상 FCFS보다 좋은가?**
A: 아니요. 요청 길이의 분산이 클 때 효과적입니다. 모든 요청이 비슷한 길이라면 차이가 적을 수 있습니다.

**Q: 예측 업데이트 간격을 바꿀 수 있나?**
A: 네, `VLLM_ELIS_PREDICTION_INTERVAL` 환경 변수로 조정 가능합니다.

**Q: 다른 예측 모델을 사용할 수 있나?**
A: 가능합니다. `ELISPredictor` 클래스와 호환되는 인터페이스를 구현하면 됩니다.

**Q: 예측이 실패하면?**
A: 예측 실패 시 `predicted_remaining_tokens = float('inf')`로 설정되어 FCFS처럼 동작합니다.

---

**작성일**: 2025.12.07  
**작성자**: hyunnnchoi  
**버전**: vLLM v0.11.0-debug + ISRTF

