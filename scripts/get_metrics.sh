#!/bin/bash

VRS=$1
EPOCH=$2
DATASET=${3:-sen2venus}

HPC=${HPC:-hpc}

IFS=$'\n '
######

CONFIG_FILE=cfgs/${DATASET}_v${VRS}.yml
OUT_DIR=output/${HPC}/${DATASET}_v${VRS}/

RESULT=`python src/main.py --phase test --config $CONFIG_FILE --output $OUT_DIR --epoch ${EPOCH} | grep PSNR -A1 |tail -n1`

echo $RESULT | awk '{print $1}'
echo $RESULT | awk '{print $2}'
echo $RESULT | awk '{print $3}'
echo $RESULT | awk '{print $4}'
echo $RESULT | awk '{print $5}'
echo $RESULT | awk '{print $6}'
