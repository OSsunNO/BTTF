#!/bin/bash
# =============================================================
# BTTF_new 단변량(univariate) 실험 실행 스크립트
#
# 위치: BTTF_new/scripts/run_univariate.sh
# 실행: 프로젝트 루트(BTTF_new/)에서 아래 명령어로 실행
#
#   bash scripts/run_univariate.sh               # 전체 실행
#   bash scripts/run_univariate.sh exchange_rate # 특정 데이터셋만
#   bash scripts/run_univariate.sh etth1 96      # 특정 데이터셋 + pred_len
#
# target_col: date 제외 후 숫자 컬럼 기준 0-indexed
#   exchange_rate → OT = 7번 (컬럼: 0,1,2,3,4,5,6,OT)
#   etth1         → OT = 6번 (컬럼: HUFL,HULL,MUFL,MULL,LUFL,LULL,OT)
#   ettm2         → OT = 6번 (컬럼: HUFL,HULL,MUFL,MULL,LUFL,LULL,OT)
#   illness       → OT = 6번 (컬럼: %WLI,%UWLI,AGE0-4,AGE5-24,ILI,PROV,OT)
# =============================================================

set -e

# ─────────────────────────────────────────
# 프로젝트 루트 = 이 스크립트의 한 단계 위 디렉토리
# ─────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
PYTHON="python3"

# ─────────────────────────────────────────
# 공통 하이퍼파라미터
# ─────────────────────────────────────────
EPOCHS1=1
EPOCHS2=1
SEED=0
DEVICE="cuda"      # GPU 없으면 자동으로 cpu 전환됨
NUM_WORKERS=0

SEG_MODE="div"
SEG_DIV=3
SEG_STRIDE=1

K_STEP=5
ALPHA=1.0
BETA=1.0
MODE_SPLIT="test"

# ─────────────────────────────────────────
# 필터 인자
# ─────────────────────────────────────────
FILTER_DS="${1:-all}"
FILTER_P="${2:-all}"

# ─────────────────────────────────────────
# 실행 함수
# ─────────────────────────────────────────
run_exp() {
    local DATASET=$1
    local PRED_LEN=$2
    local TARGET_COL=$3

    if [ "$FILTER_DS" != "all" ] && [ "$FILTER_DS" != "$DATASET" ]; then
        return 0
    fi
    if [ "$FILTER_P" != "all" ] && [ "$FILTER_P" != "$PRED_LEN" ]; then
        return 0
    fi

    echo ""
    echo "============================================================"
    echo " Dataset=${DATASET}  pred_len=${PRED_LEN}  target_col=${TARGET_COL}"
    echo "============================================================"

    cd "${PROJECT_ROOT}"
    ${PYTHON} -m scripts.run \
        --dataset       "${DATASET}" \
        --pred_len      "${PRED_LEN}" \
        --target_col    "${TARGET_COL}" \
        --root_dir      "${OUTPUT_DIR}" \
        --seed          "${SEED}" \
        --device        "${DEVICE}" \
        --epochs1       "${EPOCHS1}" \
        --epochs2       "${EPOCHS2}" \
        --num_workers   "${NUM_WORKERS}" \
        --seg_mode      "${SEG_MODE}" \
        --seg_div       "${SEG_DIV}" \
        --seg_stride    "${SEG_STRIDE}" \
        --k_step        "${K_STEP}" \
        --alpha         "${ALPHA}" \
        --beta          "${BETA}" \
        --mode_split    "${MODE_SPLIT}" \
        --scale \
        --stage1_tag    "stage1" \
        --stage2_tag    "stage2"
    # batch_size, lr1, lr2, patience 생략 → dataset_configs의 pred_len별 값 자동 적용
}

# ─────────────────────────────────────────
# 실험 목록
# ─────────────────────────────────────────

# exchange_rate: 숫자 컬럼 [0~7], OT=7
for P in 96 192 336 720; do
    run_exp "exchange_rate" "${P}" 7
done

# etth1: 숫자 컬럼 [0~6], OT=6
for P in 96 192 336 720; do
    run_exp "etth1" "${P}" 6
done

# ettm2: 숫자 컬럼 [0~6], OT=6
for P in 96 192 336 720; do
    run_exp "ettm2" "${P}" 6
done

# illness: 숫자 컬럼 [0~6], OT=6
for P in 24 36 48 60; do
    run_exp "illness" "${P}" 6
done

echo ""
echo "=============================="
echo " 모든 실험 완료"
echo "=============================="
