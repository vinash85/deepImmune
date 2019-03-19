#!/bin/bash
#SBATCH -c 2                               # 1 core
#SBATCH -t 0-02:10                         # Runtime of 5 minutes, in D-HH:MM format
#SBATCH -p short                           # Run in short partition
#SBATCH -o hostname_%j.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
#SBATCH --mail-type=ALL                    # ALL email notification type
#SBATCH --mail-user=kompa@fas.harvard.edu  # Email to which notifications will be sent
rm -rf venv2712/

module load gcc/6.2.0 
module load python/2.7.12
virtualenv venv2712 --system-site-packages

source venv2712/bin/activate 
pip install -U numpy 
pip install -r requirements_2_7_12.txt
pip install --no-cache-dir torch
pip install --no-cache-dir torchvision
pip install --no-cache-dir tensorboardX

deactivate
