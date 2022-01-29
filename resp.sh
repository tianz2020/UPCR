#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=Response
#SBATCH -o ./response.%A.out
#SBATCH -e ./response.%A.err
#SBATCH -p debug
#SBATCH --gres=gpu:$2
#SBATCH -c2
#SBATCH --mem=40G
#SBATCH --time=20-00:00:00
#SBATCH --nodelist=$1

# Set-up the environment.
source ${HOME}/.bashrc
conda activate mywork

set PYTHONPATH=./

# Start the experiment. CUDA_VISIBLE_DEVICES=$6
# CUDA_VISIBLE_DEVICES=$3
python main_resp.py
EOT
