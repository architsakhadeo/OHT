#!/bin/sh

#SBATCH --account=
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=

chmod +x task*
cd ../Experiments/
module load python/3.6
source $HOME/your_env_name/bin/activate
'../Script/tasks_'"$SLURM_ARRAY_TASK_ID"'.sh'
