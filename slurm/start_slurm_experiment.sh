#!/bin/bash

#PARTITION='edison5a'
PARTITION1='edison5a'
PARTITION2='edison5b'

#JOB_NAME='ssimon-config-space'
JOB_NAME1='ssimon-config-space-part1'
JOB_NAME2='ssimon-config-space-part2'

#COMMANDS_FILE="./test.txt"
COMMANDS_FILE='./projects.txt'
#COMMANDS_FILE='./commands.txt'

# Split the commands file
head -n 1000 $COMMANDS_FILE > commands_part1.txt
tail -n +1001 $COMMANDS_FILE | head -n 1000 > commands_part2.txt


NTASKS1=$(cat commands_part1.txt | wc -l)
NTASKS2=$(cat commands_part2.txt | wc -l)

echo "NTASK1: $NTASKS1"
echo "NTASK2: $NTASKS2"

echo "$(date) Submitting job $JOB_NAME1 to $PARTITION1 monitoring $NTASKS1 projects."
echo "$(date) Submitting job $JOB_NAME2 to $PARTITION2 monitoring $NTASKS2 projects."

mkdir -p /home/ssimon/logs

#NTASKS=$(cat $COMMANDS_FILE | wc -l)
#echo "$(date) Submitting job $JOB_NAME to $PARTITION monitoring $NTASKS configurations."
#mkdir -p /home/ssimon/logs

#sbatch --error="/home/ssimon/logs/%x_%A_%a.err" --output="/home/ssimon/logs/%x_%A_%a.out" --exclusive --array=1-$NTASKS --partition=$PARTITION --job-name=$JOB_NAME ./analysis_job.sh $COMMANDS_FILE

sbatch --error="/home/ssimon/logs/%x_%A_%a.err" --output="/home/ssimon/logs/%x_%A_%a.out" --exclusive --array=1-$NTASKS1 --partition=$PARTITION1 --job-name=$JOB_NAME1 ./analysis_job.sh commands_part1.txt
sbatch --error="/home/ssimon/logs/%x_%A_%a.err" --output="/home/ssimon/logs/%x_%A_%a.out" --exclusive --array=1-$NTASKS2 --partition=$PARTITION2 --job-name=$JOB_NAME2 ./analysis_job.sh commands_part2.txt