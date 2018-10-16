#!/bin/bash

wkdir=$(pwd)

event_list=${1:?[arg]need event_list}
out_dir=${2:?[arg]out_dir}

out_dir=$(readlink -f $out_dir)

#for event_id in $(awk -F"|" 'NF&&$1!~/#/{print $9}' $event_list)
for event_id in $(awk -F"|" 'NF&&$1!~/#/{printf "%s.%s.%s.%s\n", $1,$2,$3,$4}' $event_list)
do

  echo "====== $event_id"
  cd $wkdir/$event_id/misfit

  for win in $(awk '$1!~/#/{print $2}' misfit.txt | sort -u )
  do
    echo $win
    output_fig=$out_dir/${event_id}_${win}.pdf
    input_figs=$(ls figure/*_${win}.pdf)
    err=$?
    echo $input_figs
    if [ $err -ne 0 ]
    then
      echo "[ERROR] cannot find input figures"
    else
      #rm $output_fig
      pdf_merge.sh $output_fig $input_figs
    fi
    #exit -1
  done

done