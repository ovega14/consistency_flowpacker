#!/bin/bash
#SBATCH --account=beqv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=run_rcd_pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ../venv/bin/activate
export PATH="/u/octavio5/projects/consistency_flowpacker/venv/bin:$PATH"
cd /u/octavio5/projects/consistency_flowpacker/scripts

# Training args
#MODEL_TYPE="MPConsistencyModel"
#MODEL_TYPE="ConditionedMPConsistencyModel"
MODEL_TYPE="ConditionedMPConsistencyModelV2"
EPOCHS=500
BATCH_SIZE=64
LR=1e-3
EMA_MU=0.99
SCHEDULER="ShadowScheduler"

# Sampling args
SAVE_DIR="../checkpoints/${MODEL_TYPE}_ep${EPOCHS}_lr${LR}_bs${BATCH_SIZE}"
SEED=42

python3 ../experiments/train_distillation.py \
    --traj_dir ../flowpacker/samples/traj-train/run_1 \
    --model_type $MODEL_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --ema_mu $EMA_MU \
    --scheduler $SCHEDULER \
    --save_interval $EPOCHS \
    --save_dir $SAVE_DIR

python3 ../experiments/eval_consistency.py \
    --traj_dir ../flowpacker/samples/traj-test/run_1 \
    --ckpt_path ${SAVE_DIR}/consistency_ep${EPOCHS}.pt \
    --model_type $MODEL_TYPE \
    --save_dir $SAVE_DIR \
    --seed $SEED
