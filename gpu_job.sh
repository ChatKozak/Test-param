#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=gpu  # Use the 'gpu' partition
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4  # Adjust this based on your requirements
#SBATCH --time=00:10:00  # Adjust this to your desired runtime

# Load any necessary modules or environment variables for your GPU setup
# For example, loading CUDA if your script requires it
# module load cuda  # Uncomment this line if CUDA is required

# Navigate to the directory containing your RML.py script
cd /path/to/your/script/directory

# Run your Python script
python RML.py
