PARAM="$1"
DRY_RUN=""
if [ "$PARAM" = "DRY_RUN" ]; then
	echo "Using DRY RUN"
	DRY_RUN="1"
fi

# List of tuple with (env, mode)
declare -a base_mt10_envs=(
    "Phoenix-v5"
    "Assault-v5"
    "DemonAttack-v5"
    "AirRaid-v5"
    "SpaceInvaders-v5"
    "Carnival-v5"
)

# List of tuple with (env, mode)
declare -a target_mt10_envs=(
    "Phoenix-v5 0"
    "Assault-v5 0"
    "DemonAttack-v5 1"
    "AirRaid-v5 1"
    "SpaceInvaders-v5 0"
    "Carnival-v5 0"
)

worker_id=0

DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="./slurm/logs/slurm_${DATE}_matrix"
echo "SLURM Log directory: ${LOGDIR}"
mkdir -p "$LOGDIR"
mkdir -p "$LOGDIR/worker_scripts"
mkdir -p "$LOGDIR/worker_logs"


for base_env in "${base_mt10_envs[@]}"; do
    for target_env in "${target_mt10_envs[@]}"; do
        read -a strarr <<< "$target_env"  # uses default whitespace IFS

        echo "Running ${base_env} -> ${strarr[0]}"

        WORKER_SCRIPT="$LOGDIR/worker_scripts/worker-$worker_id-$base_env-${strarr[0]}.slurm"
	    WORKER_OUT="$LOGDIR/worker_logs/slurm-worker-$worker_id-$base_env-${strarr[0]}.out"

        worker_id=$((${worker_id} + 1))
        echo "\
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=${base_env}-${strarr[0]}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output $WORKER_OUT
#SBATCH --error $WORKER_OUT

echo \"========== SLURM JOB INFO ==========\"
echo
echo \"The job will be started on the following node(s):\"
echo $SLURM_JOB_NODELIST
echo
echo \"Slurm User:         $SLURM_JOB_USER\"
echo \"Run Directory:      $(pwd)\"
echo \"Job ID:             $SLURM_JOB_ID\"
echo \"Job Name:           $SLURM_JOB_NAME\"
echo \"Partition:          $SLURM_JOB_PARTITION\"
echo \"Number of nodes:    $SLURM_JOB_NUM_NODES\"
echo \"Number of tasks:    $SLURM_NTASKS\"
echo \"Submitted From:     $SLURM_SUBMIT_HOST\"
echo \"Submit directory:   $SLURM_SUBMIT_DIR\"
echo \"Hostname:           $(hostname)\"
echo
echo \"Dashboard Host:     |$(hostname):8787|\"
echo

echo
echo \"========== Start ==========\"
date

echo
echo \"========== Setup ==========\"


echo
echo \"========== Starting python script ==========\"
python atari_experiment/ppo_atari_transfer.py -b ALE/"${base_env}" -t ALE/"${strarr[0]}"

echo
echo \"========== Done ==========\"
date" >"$WORKER_SCRIPT"
        if [ -z "$DRY_RUN" ]; then sbatch "$WORKER_SCRIPT"; fi
    done
done
