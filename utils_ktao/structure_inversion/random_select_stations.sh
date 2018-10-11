#!/bin/bash

# structure inversion 

#------ read command line args
control_file=${1:?[arg] need control_file}
outfile=${2:?[arg]output event_list}

#------ source control_file
source $control_file

$sem_utils_dir/structure_inversion/random_select_stations.py \
  $data_dir/data/STATIONS.REF_ENU ${minimum_inter_station_distance} "STATIONS.REF_ENU.subset" "STATIONS.REF_ENU.subset.pdf"

awk '{printf "%s.%s\n",$1,$2}' STATIONS.REF_ENU.subset > grep.f
grep -f grep.f $data_dir/data/event.txt > ${outfile}

$sem_utils_dir/structure_inversion/random_select_stations.py \
  STATIONS.REF_ENU.subset ${minimum_inter_station_distance_for_common_set} \
  "STATIONS.REF_ENU.subset.include" "STATIONS.REF_ENU.subset.include.pdf"

grep -vF -f STATIONS.REF_ENU.subset.include STATIONS.REF_ENU.subset > STATIONS.REF_ENU.subset.exclude 

$sem_utils_dir/structure_inversion/random_select_stations_with_include_and_exclude.py \
  $data_dir/data/STATIONS.REF_ENU STATIONS.REF_ENU.subset.include STATIONS.REF_ENU.subset.exclude \
  ${minimum_inter_station_distance} \
  "STATIONS.REF_ENU.subset.new" "STATIONS.REF_ENU.subset.new.pdf"