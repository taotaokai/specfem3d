#!/bin/bash

# structure inversion 

#------ read command line args
control_file=${1:?[arg] need control_file}
outfile=${2:?[arg]output event_list}

#------ source control_file
source $control_file

$sem_utils_dir/structure_inversion/random_select_stations.py \
  $data_dir/data/STATIONS.REF_ENU ${minimum_inter_station_distance} "STATIONS.REF_ENU.select" "STATIONS.REF_ENU.select.pdf"

awk '{printf "%s.%s\n",$1,$2}' STATIONS.REF_ENU.select > grep.f
grep -f grep.f $data_dir/data/event.txt > ${outfile}