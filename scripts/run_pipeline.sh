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

python3 train_distillation.py \
    --traj_dir ../flowpacker/samples/traj-500/run_1 \
    --model_type ConditionedMPConsistencyModel \
    --epochs 200 \
    --batch_size 64 \
    --lr 1e-3 \
    --ema_mu 0.99 \
    --save_interval 20 \
    --save_dir ../checkpoints/consistency

python3 eval_consistency.py \
    --traj_dir ../flowpacker/samples/traj-500/run_1 \
    --ckpt_path ../checkpoints/consistency/consistency_ep200.pt \
    --model_type ConditionedMPConsistencyModel \
    --n_test 100 \
    --seed 42
