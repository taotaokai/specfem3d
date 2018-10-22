#!/bin/bash

# structure inversion 

#------ read command line args
control_file=${1:?[arg] need control_file}
#prev_subset=${2:?[arg]e.g. STATIONS.REF_ENU.subset}

#------ source control_file
source $control_file

# get common set
prev_subset=$prev_iter_dir/STATIONS.REF_ENU.subset
${python_exec} $sem_utils_dir/structure_inversion/random_select_stations.py \
  ${prev_subset} ${minimum_inter_station_distance_for_common_set} \
  "STATIONS.REF_ENU.subset.common" "STATIONS.REF_ENU.subset.common.pdf"

grep -vF -f STATIONS.REF_ENU.subset.common ${prev_subset} | grep -v ^# > STATIONS.REF_ENU.subset.exclude 

awk '$1!~/#/{printf "%s.%s\n",$1,$2}' STATIONS.REF_ENU.subset.common > grep.f
grep -f grep.f $data_dir/data/event.txt > event_common.txt

# get a subset with common set included
${python_exec} $sem_utils_dir/structure_inversion/random_select_stations_with_include_and_exclude.py \
  $data_dir/data/STATIONS.REF_ENU STATIONS.REF_ENU.subset.common STATIONS.REF_ENU.subset.exclude \
  ${minimum_inter_station_distance} \
  "STATIONS.REF_ENU.subset" "STATIONS.REF_ENU.subset.pdf"

awk '$1!~/#/{printf "%s.%s\n",$1,$2}' STATIONS.REF_ENU.subset > grep.f
grep -f grep.f $data_dir/data/event.txt > event.txt
rm grep.f