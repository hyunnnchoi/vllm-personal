# ISRTF 스케줄러 + ELIS 예측 모델 빠른 시작 가이드

## 🎯 목표

vLLM 스케줄러에 ISRTF (Iterative Shortest Remaining Time First) 정책과 ELIS 예측 모델을 통합하여 요청 latency를 최적화합니다.

## ✅ 현재 상태

모든 구성 요소가 준비되었습니다!

```
✓ ELIS 예측 모델 체크포인트 (499MB)
✓ ELIS 모델 코드
✓ vLLM 스케줄러 ISRTF 정책
✓ ISRTFRequestQueue 구현
✓ 스케줄러에 ELIS 예측기 통합
✓ Request 클래스 수정
✓ 테스트 스크립트
✓ Request generator
```

## 🚀 빠른 시작

### 1️⃣ 상태 확인

```bash
cd /home/xsailor6/hmchoi
python3 check_isrtf_status.py
```

모든 체크가 통과하면 ✅ 테스트 준비 완료!

### 2️⃣ ISRTF 서버 실행

```bash
cd /home/xsailor6/hmchoi/vllm-deployments/vanilla-vllm
./vanilla_vllm_isrtf.sh
```

**주요 설정:**
- 정책: `--scheduling-policy isrtf`
- 모델: `upstage/SOLAR-10.7B-v1.0`
- TP: 4 (Tensor Parallelism)
- GPU memory: 80%

### 3️⃣ 로그 확인

서버가 시작되면 ELIS 예측 모델이 로드됩니다:

```bash
sudo docker logs vllm 2>&1 | grep ELIS
```

기대 출력:
```
[ELIS] Initializing predictor...
[ELIS] Checkpoint: /ELIS/train/checkpoints/latest_model.pt
[ELIS] BGE model: BAAI/bge-base-en-v1.5
[ELIS] Predictor initialized successfully on cuda
```

### 4️⃣ 테스트 요청 전송

새 터미널에서:

```bash
cd /home/xsailor6/hmchoi/vllm/kukt/request_generator

# 간단한 테스트
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "upstage/SOLAR-10.7B-v1.0",
    "prompt": "Explain machine learning in one paragraph.",
    "max_tokens": 200
  }'
```

### 5️⃣ FCFS vs ISRTF 성능 비교

전체 테스트 스크립트 실행:

```bash
cd /home/xsailor6/hmchoi
./test_isrtf.sh
```

이 스크립트는:
1. FCFS baseline 테스트 실행
2. ISRTF 테스트 실행
3. 결과 비교 및 저장

결과는 `/home/xsailor6/hmchoi/ELIS/data/test_results/`에 저장됩니다.

## 📊 예측 모델 동작 방식

1. **초기 예측**: 새 요청이 들어오면 프롬프트 텍스트로 예측
2. **우선순위 설정**: `predicted_remaining_tokens`가 작을수록 높은 우선순위
3. **업데이트**: 50토큰 생성마다 예측 재계산 (설정 가능)
4. **재정렬**: 우선순위 큐에서 자동으로 순서 조정

## 🔧 환경 변수 설정

`vanilla_vllm_isrtf.sh`에서 설정 가능:

```bash
# ELIS 체크포인트 경로
ELIS_CHECKPOINT="/ELIS/train/checkpoints/latest_model.pt"

# BGE 임베딩 모델
ELIS_BGE_MODEL="BAAI/bge-base-en-v1.5"

# 예측 업데이트 간격 (토큰 수)
ELIS_PREDICTION_INTERVAL=50

# ELIS 모듈 경로
ELIS_PATH="/ELIS"
```

## 📈 성능 모니터링

### 실시간 로그 확인

```bash
# ELIS 예측 로그
sudo docker logs -f vllm 2>&1 | grep "ELIS"

# 스케줄링 결정
sudo docker logs -f vllm 2>&1 | grep "ISRTF"

# 전체 로그
sudo docker logs -f vllm
```

### 메트릭 확인

```bash
# Prometheus 메트릭
curl http://localhost:8000/metrics

# 모델 정보
curl http://localhost:8000/v1/models
```

## 🎨 테스트 시나리오

### 시나리오 1: 혼합 길이 요청

```json
{
  "prompts": [
    "짧은 요청 (예상 50 tokens)",
    "중간 요청 (예상 150 tokens)",
    "긴 요청 (예상 400 tokens)"
  ]
}
```

**기대 결과**: ISRTF가 짧은 요청을 우선 처리 → 평균 latency 감소

### 시나리오 2: 동시 요청

```bash
# 배치 크기 조정
python send_to_vllm.py --batch-size 10
```

**기대 결과**: 높은 동시성에서 ISRTF의 효과가 더 명확

## 📝 디버깅

### 문제: 서버가 시작되지 않음

```bash
# 로그 확인
sudo docker logs vllm

# 컨테이너 상태
sudo docker ps -a | grep vllm

# GPU 확인
nvidia-smi
```

### 문제: ELIS 모델이 로드되지 않음

```bash
# 파일 존재 확인
ls -lh /home/xsailor6/hmchoi/ELIS/train/checkpoints/latest_model.pt

# 컨테이너 마운트 확인
sudo docker exec -it vllm ls -la /ELIS/train/checkpoints/
```

### 문제: 예측이 작동하지 않음

```bash
# 정책 확인
sudo docker logs vllm 2>&1 | grep "policy"

# 예측 로그
sudo docker logs vllm 2>&1 | grep "prediction" | tail -20
```

## 📚 추가 자료

- **상세 가이드**: `ISRTF_TEST_GUIDE.md`
- **ELIS 논문**: https://arxiv.org/abs/2505.09142
- **커밋 히스토리**: `git log --oneline | head -20`

## 🔥 빠른 명령어 모음

```bash
# 1. 상태 확인
python3 check_isrtf_status.py

# 2. ISRTF 서버 시작
cd vllm-deployments/vanilla-vllm && ./vanilla_vllm_isrtf.sh

# 3. 로그 모니터링 (새 터미널)
sudo docker logs -f vllm 2>&1 | grep -E "ELIS|ISRTF"

# 4. 테스트 요청 (새 터미널)
cd vllm/kukt/request_generator
python send_to_vllm.py --batch-size 5

# 5. 서버 중지
sudo docker stop vllm && sudo docker rm vllm

# 6. 전체 테스트
./test_isrtf.sh
```

## ✨ 다음 단계

1. ✅ 기본 ISRTF 동작 확인
2. ⏱️ FCFS vs ISRTF 성능 비교
3. 🔧 하이퍼파라미터 튜닝 (예측 간격, GPU 메모리 등)
4. 📊 대규모 워크로드 테스트
5. 🎯 예측 정확도 분석 및 모델 개선

---

**문제가 있으면**: 로그를 확인하고 `ISRTF_TEST_GUIDE.md`의 디버깅 섹션을 참고하세요!

