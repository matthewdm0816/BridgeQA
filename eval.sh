#!/bin/bash
#SBATCH --job-name=scanqaeval   # create a short name for your job

##SBATCH --partition=gpu   # specify the partition name: gpu 
##SBATCH --qos=gpu
##SBATCH --account=research

#SBATCH --qos=lv4
#SBATCH --time=10:00:00 
##SBATCH --account=research

#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300G               # total memory (RAM) per node

#SBATCH --cpus-per-task=24    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1  # number of gpus per node
#SBATCH --output=logs/predict-%j.out  # output format
#SBATCH --error=logs/predict-%j.out  # error output file

# Workding Dir.
cd /home/mowentao/data/ScanQA-cleaned/

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $SLURM_JOB_GPUS | tr -cd , | wc -c)+1))
echo $SLURM_GPUS

# Proxy
export PORT=$(shuf -i 2000-3000 -n 1)

## Unicycle Training
export all_proxy='http://10.141.0.110:17893'
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
    

stdbuf -o0 -e0 torchrun --nproc_per_node=$SLURM_GPUS --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
    scripts/predict.py \
    --folder "/scratch/generalvision/mowentao/ScanQA-cleaned/outputs/2023-12-18_05-46-37_ALLANSWER" \
    --i2tfile "/scratch/generalvision/mowentao/ScanQA-cleaned/scene_eval_scanqa_decl_qonly_gpt3.5_reimpl.pkl" \
    --test_type test_w_obj --batch_size 2 \
    # --folder "/scratch/generalvision/mowentao/ScanQA/outputs/2023-08-13_15-53-03_ALLANSWER/"  \
    # --i2tfile "/scratch/mowentao/BLIP/scene_eval_scanqa_interrogative_video.pkl" 
    # --midfix "attentions" # --open_ended \


