#!/bin/bash
#SBATCH --job-name=process_results
#SBATCH --output=process_results_%j.out
#SBATCH --error=process_results_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load anaconda3/2024.6
python process_simulation_results.py