#!/bin/bash
#SBATCH -JEsaTsTrojanHorseHunt            # Job name
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=32G                 # Memory per core
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs
#SBATCH -t180                             # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=tweeratunga3@gatech.edu # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                     # Change to working directory

module load cuda

uv sync
source .venv/bin/activate

srun python fourierdl.py                      # Example Process
