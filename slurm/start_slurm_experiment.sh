#!/bin/bash

PARTITION='edison5b'
JOB_NAME='ssimon-config-space'

COMPANY='disney'
COMMANDS_FILE='../data/disney/disney_commands.txt'

NTASKS=$(cat $COMMANDS_FILE | wc -l)
echo "$(date) Submitting job $JOB_NAME to $PARTITION monitoring $NTASKS tasks."
mkdir -p /home/ssimon/logs

sbatch \
  --error="/home/ssimon/logs/%x_%A_%a.err" \
  --output="/home/ssimon/logs/%x_%A_%a.out" \
  --exclusive \
  --array=1-$NTASKS \
  --partition=$PARTITION \
  --job-name=$JOB_NAME \
  --time=31-00:00:00 \
  ./analysis_job.sh $COMPANY $COMMANDS_FILE 
