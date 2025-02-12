#!/bin/bash

COMMANDS_FILE=$1
TASK_ID=$SLURM_ARRAY_TASK_ID
JOB_ID=$SLURM_ARRAY_JOB_ID
SB_JOB_NAME=$SBATCH_JOB_NAME
SLURM_JOB_NAME=$SLURM_JOB_NAME

hostname
echo "$USER"

# prepare filesystem
rm -rf /tmp/$USER/*
#mkdir -p /tmp/$USER/current_repo
#mkdir -p /tmp/$USER/results

## PYTHON environment with energy metering
rm -rf /tmp/$USER/py_env
DIR_ENV="/tmp/$USER/py_env"
mkdir -p $DIR_ENV
if [ -d "$DIR_ENV" ]; then
  if [ "$(ls -A $DIR_ENV)" ]; then
    echo "sourcing python environment"
    source $DIR_ENV/repo_env/bin/activate
  else
    echo "installing and sourcing python environment."

    #creating virtualenv
    python3 -m virtualenv $DIR_ENV/repo_env
    source $DIR_ENV/repo_env/bin/activate

    # install energy measurement scripts
    #  TODO: should be replaced wit public git repository
    python3 -m pip install /home/$USER/XXXXXXProjekt/ --no-cache-dir
  fi
else
  echo "Directory $DIR_ENV not found."
fi


## Experiment data
DIR_EXP="/tmp/$USER/experiment"
rm -rf $DIR_EXP
mkdir -p $DIR_EXP
#mkdir -p /tmp/$USER/experiment/x264
if [ -d "$DIR_EXP" ]; then
  if [ "$(ls -A $DIR_EXP)" ]; then
    echo "$(ls -A $DIR_EXP)"
    echo "dir ready for experiments"
  else
    # clone the repository
    echo "dir is empty."
    #cp -rv ./systems/x264/x264 $DIR_EXP/
    #cp -rv ./systems/artificial/artificial $DIR_EXP/
    #cp -rv ./exec_bash.py $DIR_EXP/exec_bash.py
    #cp -rv ./systems/x264/sintel_trailer_2k_480p24.y4m $DIR_EXP/
  fi
else
	echo "Directory $DIR_EXP not found."
fi
mkdir -p /tmp/$USER/experiment/output


## Execute Job
COMMAND=$(sed "${TASK_ID}q;d" $COMMANDS_FILE)

START=$(date +%s%3N)
$COMMAND
END=$(date +%s%3N)
echo "time spent on task: $(($END - $START))"


echo 'copy results'
# copy results from tmp directory to home
OUT_FOLDER="/home/$USER/results/$HOSTNAME/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p $OUT_FOLDER
cp -rv /tmp/$USER/energy $OUT_FOLDER
cp -rv /tmp/$USER/performance $OUT_FOLDER

cp -rv "/home/$USER/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" $OUT_FOLDER
cp -rv "/home/$USER/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $OUT_FOLDER


# deactivate virtualenv
deactivate
rm -rf /tmp/$USER/*
echo "canceling ${JOB_ID}_${TASK_ID}"
scancel "${JOB_ID}_${TASK_ID}"
