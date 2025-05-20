#!/bin/bash

COMMANDS_FILE=$1
TASK_ID=$SLURM_ARRAY_TASK_ID
JOB_ID=$SLURM_ARRAY_JOB_ID
SB_JOB_NAME=$SBATCH_JOB_NAME
SLURM_JOB_NAME=$SLURM_JOB_NAME

hostname

# Extract command from the file
COMMAND=$(sed "${TASK_ID}q;d" $COMMANDS_FILE)
PROJECT_NAME=$(echo "$COMMAND" | grep -oP '(?<=--url=)[^ ]+' | sed 's|.*/||')

# Define output directory and expected result file
OUT_FOLDER="/home/ssimon/GitHub/config-space/slurm/microservices/${PROJECT_NAME}"
RESULT_FILE="${OUT_FOLDER}/${PROJECT_NAME}.json"

# Check if output folder and result file exist
if [ -f "$RESULT_FILE" ]; then
  echo "Result file already exists: $RESULT_FILE"
  echo "Skipping job execution."
  exit 0
fi


# prepare filesystem
rm -rf /tmp/ssimon/config-space
mkdir -p /tmp/ssimon/config-space

## PYTHON environment
DIR_EXP="/tmp/ssimon/config-space/experiments"
rm -rf $DIR_EXP
mkdir -p $DIR_EXP
if [ -d "$DIR_EXP" ]; then
  if [ "$(ls -A $DIR_EXP)" ]; then
    echo "sourcing python environment"
    source $DIR_EXP/enery_env/bin/activate
  else
    echo "installing and sourcing python environment."

    # creating virtualenv
    #python3 -m pip install virtualenv
    python3 -m venv $DIR_EXP/env
    #python3 -m virtualenv $DIR_EXP/env
    source $DIR_EXP/env/bin/activate

    python3 --version
    python3 -m pip install git+https://github.com/AI-4-SE/CfgNet.git@config-space
    python3 -m pip install tqdm
  fi
else
  echo "Directory $DIR_EXP not found."
fi

## Experiment data
if [ -d "$DIR_EXP" ]; then
  if [ "$(ls -A $DIR_EXP)" ]; then
    # clone the repository
    echo "Clone script"
    cp -rv /home/ssimon/GitHub/config-space/src/analysis.py $DIR_EXP/analysis.py
  fi
else
	echo "Directory $DIR_EXP not found."
fi

## Execute Job
echo "Project: $PROJECT_NAME"
echo "Current directory: $(pwd)"

START=$(date +%s%3N)
$COMMAND
END=$(date +%s%3N)
echo "time spent on task: $(($END - $START))"

# copy results from tmp directory to home
echo 'copy results'
mkdir -p $OUT_FOLDER
cp -rv "/tmp/ssimon/config-space/experiments/${PROJECT_NAME}.json" $OUT_FOLDER
cp -rv "/home/ssimon/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" $OUT_FOLDER
cp -rv "/home/ssimon/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $OUT_FOLDER

# deactivate virtualenv
deactivate
rm -rf /tmp/ssimon/*
echo "canceling ${JOB_ID}_${TASK_ID}"
scancel "${JOB_ID}_${TASK_ID}"
