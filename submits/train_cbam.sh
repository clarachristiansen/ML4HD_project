#!/bin/sh 
### General options 
### -- specify queue -- 
# BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J train_cbam
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 05:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s214659@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /work3/s214659/ML4HD/ML4HD_project/submits/outputs/train_cbam%J.out
#BSUB -e /work3/s214659/ML4HD/ML4HD_project/submits/outputs/train_cbam%J.err

### Conda and cwd
cd /work3/s214659/ML4HD/ML4HD_project || exit 1
source /work3/s214659/miniconda3/etc/profile.d/conda.sh
conda activate ML4HD

### Get wandb settings
set -a
. config/.env
set +a

ARCHITECTURE="cnn_tpool2_cbam"
FRAMES=98
FINAL_DATA="mfccs"

### Run
python -m src.train --use_wandb \
                    --wandb_run_name "train_${ARCHITECTURE}_${FINAL_DATA}_frames${FRAMES}_${LSB_JOBID}" \
                    --architecture ${ARCHITECTURE} \
                    --epochs 25 \
                    --frames ${FRAMES} \
                    --final_data ${FINAL_DATA}