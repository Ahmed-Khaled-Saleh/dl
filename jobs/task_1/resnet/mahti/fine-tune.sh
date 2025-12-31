#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=task1_dl_fine_tune
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err


module --force purge
module load pytorch
source /projappl/project_2009050/mytorch/bin/activate
cd /projappl/project_2009050/dl/
pip install -e .

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/mytorch/lib/python3.11/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"


ts=$(date +%Y%m%d_%H%M%S)
srun python main.py --config ./cfgs/task_1/resnet/fine-tuning.yaml --env_file ./.env --timestamp ${ts}
