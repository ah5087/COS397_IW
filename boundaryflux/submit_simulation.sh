#!/bin/bash
#SBATCH --job-name=process_cell_migration
#SBATCH --output=output_processing.log
#SBATCH --error=error_processing.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=normal

module load anaconda3/2024.6
python3 process_simulation_results.py