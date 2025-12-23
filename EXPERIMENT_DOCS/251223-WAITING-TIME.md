# ============================================================
# SECTION 1: vLLM 서버 시작 (A100 80G * 2 환경)
# ============================================================
# NOTE, hyunnnchoi, 2025.12.23 - 벤치마크 파라미터 및 타임스탬프 설정
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_CLIENTS=128
MAX_ACTIVE_CONV=128

# NOTE, hyunnnchoi, 2025.12.23 - 서버용 decode timings 디렉토리 설정 (타임스탬프 포함)
SERVER_DECODE_TIMINGS_DIR=/home/work/hyunmokchoi/multi_turn/results/server_logs_${TIMESTAMP}_c${NUM_CLIENTS}_mac${MAX_ACTIVE_CONV}
mkdir -p "$SERVER_DECODE_TIMINGS_DIR"
export VLLM_DECODE_TIMINGS_DIR="$SERVER_DECODE_TIMINGS_DIR"

vllm serve /home/work/huggingface/huggingface/gpt-oss-20b \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --dtype auto

# ============================================================
# SECTION 2: 데이터 준비 (최초 1회만 실행)
# ============================================================
pip3 install pandas
cd /vllm/benchmarks/multi_turn
python3 convert_sharegpt_to_openai.py /home/work/hyunmokchoi/multi_turn/sharegpt.json /home/work/hyunmokchoi/multi_turn/sharegpt_conv_1024.json --seed=99 --max-items=1024

# ============================================================
# SECTION 3: 벤치마크 실행 (컨테이너 내부에서)
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_CLIENTS=128
MAX_ACTIVE_CONV=128

cd /vllm
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
cd /vllm/benchmarks/multi_turn

# 모델 경로 설정
export MODEL_PATH=/home/work/huggingface/huggingface/gpt-oss-20b

# NOTE, hyunnnchoi, 2025.12.23 - 로그 디렉토리 설정 (타임스탬프는 서버 시작 시 설정한 값 사용)
export VLLM_SCHEDULER_CSV_LOG="1"
export VLLM_SCHEDULER_CSV_LOG_DIR="/home/work/hyunmokchoi/multi_turn/results/vllm_scheduler_logs"
LOG_FILE="/home/work/hyunmokchoi/multi_turn/results/benchmark_${TIMESTAMP}_c${NUM_CLIENTS}_mac${MAX_ACTIVE_CONV}.log"

# NOTE, hyunnnchoi, 2025.12.23 - 벤치마크용 디렉토리 생성
mkdir -p "$VLLM_SCHEDULER_CSV_LOG_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# 벤치마크 실행
python3 benchmark_serving_multi_turn.py \
  --model "$MODEL_PATH" \
  --served-model-name gpt-oss-20b \
  --input-file /home/work/hyunmokchoi/multi_turn/sharegpt_conv_1024.json \
  --num-clients $NUM_CLIENTS \
  --max-active-conversations $MAX_ACTIVE_CONV \
  --excel-output \
  --verbose \
  2>&1 | tee "$LOG_FILE"

echo "=========================================="
echo "벤치마크 완료!"
echo "=========================================="
echo "Benchmark log: $LOG_FILE"
echo "Scheduler logs: $VLLM_SCHEDULER_CSV_LOG_DIR"
echo "Server decode timings: $SERVER_DECODE_TIMINGS_DIR"
echo "=========================================="

# ============================================================
# TODO: 추가할 내용
# ============================================================
# - Request 별 Waiting time, Execution time 등 
# - Request 상세 메트릭