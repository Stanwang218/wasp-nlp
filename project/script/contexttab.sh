#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-568 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1  # We're launching 2 nodes with 8 Nvidia T4 GPUs each
#SBATCH -o /mimer/NOBACKUP/groups/oovgen/ziyuan/wasp-project/wasp-nlp/project/log/contexttab.out
#SBATCH -t 0-00:30:00

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load virtualenv/20.23.1-GCCcore-12.3.0
source /mimer/NOBACKUP/groups/oovgen/ziyuan/nlp_project_env/bin/activate

python /mimer/NOBACKUP/groups/oovgen/ziyuan/wasp-project/wasp-nlp/project/tests/contexttab_tests.py