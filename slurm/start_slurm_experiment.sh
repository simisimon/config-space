#!/bin/bash

PARTITION='edison5a'
JOB_NAME='ssimon-config-space'

COMMANDS_FILE='./projects_final_updated.txt'

NTASKS=$(cat $COMMANDS_FILE | wc -l)
echo "$(date) Submitting job $JOB_NAME to $PARTITION monitoring $NTASKS configurations."
mkdir -p /home/ssimon/logs

sbatch --error="/home/ssimon/logs/%x_%A_%a.err" --output="/home/ssimon/logs/%x_%A_%a.out" --exclusive --array=1-$NTASKS --partition=$PARTITION --job-name=$JOB_NAME ./analysis_job.sh $COMMANDS_FILE
