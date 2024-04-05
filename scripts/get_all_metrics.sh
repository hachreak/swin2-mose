#!/bin/bash

VRS=$1
EPOCH_INIT=${2:-0}
DATASET=${3:-sen2venus}

HPC=${HPC:-hpc}

EPS=`ls -1 output/${HPC}/${DATASET}_v${VRS}/eval| awk -F"-" '{print $2}' | awk -F. '{print $1}'|sort -n`
# echo $EPS

echo "eval ${DATASET} v$VRS"
for EPOCH in $EPS; do
  echo "epoch $EPOCH"
  if [ $EPOCH -lt $EPOCH_INIT ]; then
      echo "skip epoch $EPOCH"
      continue
  fi
  RESULTS=`./scripts/get_metrics.sh $VRS $EPOCH $DATASET`
  # echo $RESULTS

  R_EPOCHS="$R_EPOCHS\n$EPOCH"

  R_PSNR="$R_PSNR\n`echo $RESULTS | awk '{print $1}'`"
  R_SSIM="$R_SSIM\n`echo $RESULTS | awk '{print $2}'`"
  R_CC="$R_CC\n`echo $RESULTS | awk '{print $3}'`"
  R_RMSE="$R_RMSE\n`echo $RESULTS | awk '{print $4}'`"
  R_SAM="$R_SAM\n`echo $RESULTS | awk '{print $5}'`"
  R_ERGAS="$R_ERGAS\n`echo $RESULTS | awk '{print $6}'`"
done

echo "########"
echo ""
echo "epochs"
echo -e $R_EPOCHS
echo ""
echo "psnr"
echo -e $R_PSNR
echo ""
echo "ssim"
echo -e $R_SSIM
echo ""
echo "cc"
echo -e $R_CC
echo ""
echo "rmse"
echo -e $R_RMSE
echo ""
echo "sam"
echo -e $R_SAM
echo ""
echo "ergas"
echo -e $R_ERGAS
