#!/bin/sh 
### General options 
### -- specify queue -- 
# BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J train_function_98
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 2GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00 
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
#BSUB -o train_function_98_%J.out 
#BSUB -e train_function_98_%J.err 

# here follow the commands you want to execute with input.in as the input file
cd /work3/s214659/ML4HD/ML4HD_project || exit 1
source /work3/s214659/miniconda3/etc/profile.d/conda.sh
conda activate ML4HD

python - << 'EOF'
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
EOF

python -m src.train --use_wandb --wandb_run_name "test_98"