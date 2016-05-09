#!/bin/bash
for g in `seq 0.0  0.5  18`
do
#pushd $g >>/dev/null

#for f in `seq 50 100 3000`
#do

MAE=`cat $g/errorstat.out | grep MAE | awk '{print $2}'`
RMSE=`cat $g/errorstat.out | grep RMSE | awk '{print $2}'`
echo  $g $MAE $RMSE
#done


#popd >>/dev/null

done







