#!/bin/bash
    SBATCH --job-name=my_first_sbatch   # A name for your job. Visible in squeue.
    SBATCH --nodes=1                    # Number of compute nodes to request.
    SBATCH --ntasks-per-node=1          # Tasks (processes) per node
    SBATCH --time=00:10:00              # HH:MM:SS, set a time limit for this job (here 10min)
    SBATCH --partition=debug            # Partition to use; "debug" is usually for quick tests
    # SBATCH --mem=460000                 # Memory needed (simply set the mem of a node)
    # SBATCH --cpus-per-task=288          # CPU cores per task (simply set the number of cpus a node has)
    # SBATCH --environment=my_env         # the environment to use
    SBATCH --output=/iopsstor/scratch/cscs/camillechallier/my_first_sbatch.out  # log file for stdout / prints etc
    SBATCH --error=/iopsstor/scratch/cscs/camillechallier/my_first_sbatch.err  # log file for stderr / errors

    # Exit immediately if a command exits with a non-zero status (good practice)
    set -eo pipefail

    # Print SLURM variables so you see how your resources are allocated
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Allocated Node(s): $SLURM_NODELIST"
    echo "Number of Tasks: $SLURM_NTASKS"
    echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
    echo "Current path: $(pwd)"
    echo "Current user: $(whoami)"

    source /users/camillechallier/.venv/bin/activate
    echo "using python interpreteur $(which python)"

    # Run the Python script using srun
    srun --account=a-a06 --time=10:00 -p debug python eval.py --config "example_config.yaml" --base-url "https://openrouter.ai/api/v1" --api-key 