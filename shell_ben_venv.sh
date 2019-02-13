#!/bin/bash
#SBATCH -c 2                               # 1 core
#SBATCH -t 0-00:10                         # Runtime of 5 minutes, in D-HH:MM format
#SBATCH -p short                           # Run in short partition
#SBATCH -o hostname_%j.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
#SBATCH --mail-type=ALL                    # ALL email notification type
#SBATCH --mail-user=kompa@fas.harvard.edu  # Email to which notifications will be sent
rm -rf venv1/

module load gcc/6.2.0 
module load python/3.6.0
virtualenv venv1

source venv1/bin/activate 
pip3 install -r requirements.txt

deactivate
