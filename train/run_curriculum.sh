#!/bin/bash
set -e
cd "$(dirname "$0")"

SEEDS="0 42 123"
PYTHON_SCRIPT="train_curriculum.py"

TIMESTEPS_STAGE1=500000
TIMESTEPS_STAGE2=1000000
TIMESTEPS_STAGE3=1500000

RUN_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_ROOT="training_logs_${RUN_TIMESTAMP}"
TENSORBOARD_ROOT="tensorboard_logs_${RUN_TIMESTAMP}"

echo "--- 开始完整的课程学习训练流程, 日志根目录: ${LOG_ROOT} ---"

for SEED in $SEEDS
do
    echo -e "\n>>> 开始处理 SEED ${SEED} <<<"

    # Stage 1
    echo "--- STAGE 1 (SEED ${SEED}): 正在训练“基础模型” ---"
    RUN_NAME_S1="stage1_seed${SEED}"
    FINAL_MODEL_S1="${LOG_ROOT}/${RUN_NAME_S1}/final_model.zip"
    python ${PYTHON_SCRIPT} --stage 1 --seed ${SEED} --total-timesteps ${TIMESTEPS_STAGE1} --log-folder ${LOG_ROOT} --run-name ${RUN_NAME_S1} --tensorboard-log ${TENSORBOARD_ROOT} --progress
    
    # Stage 2
    echo "--- STAGE 2 (SEED ${SEED}): 正在训练“泛化模型” ---"
    RUN_NAME_S2="stage2_seed${SEED}"
    FINAL_MODEL_S2="${LOG_ROOT}/${RUN_NAME_S2}/final_model.zip"
    python ${PYTHON_SCRIPT} --stage 2 --seed ${SEED} --total-timesteps ${TIMESTEPS_STAGE2} --log-folder ${LOG_ROOT} --run-name ${RUN_NAME_S2} --tensorboard-log ${TENSORBOARD_ROOT} --load-model-path ${FINAL_MODEL_S1} --progress

    # Stage 3
    echo "--- STAGE 3 (SEED ${SEED}): 正在训练“极限模型” ---"
    RUN_NAME_S3="stage3_seed${SEED}"
    python ${PYTHON_SCRIPT} --stage 3 --seed ${SEED} --total-timesteps ${TIMESTEPS_STAGE3} --log-folder ${LOG_ROOT} --run-name ${RUN_NAME_S3} --tensorboard-log ${TENSORBOARD_ROOT} --load-model-path ${FINAL_MODEL_S2} --progress
done

echo -e "\n--- 所有种子的完整课程学习训练已全部完成！ ---"