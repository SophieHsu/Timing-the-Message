#!/bin/bash

# Save this script as "submit_notifier.sh" and make it executable with chmod +x submit_notifier.sh

# Create the job file dynamically based on command-line arguments
ARGS="$@"

cat <<EOT > slurm/train.job
#!/bin/bash

#SBATCH --account=sophie.hsu.pi
#SBATCH --partition=fm,sdm
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=mps:50
#SBATCH --mem=16G
#SBATCH --time=72:00:00

echo "Allocated GPUs: \$CUDA_VISIBLE_DEVICES"

eval "\$(conda shell.bash hook)"
conda activate timing
echo  ${ARGS}
python src/train.py ${ARGS}
EOT

# Submit the job script using sbatch
sbatch slurm/train.job