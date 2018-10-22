#!/bin/bash

# copy files from previous iter_dir

control_file=${1:?[arg] need control_file}

#------ source control_file
source $control_file

iter_plus_one=$(echo "$iter_num" | awk '{printf "%02d", $1+1}')
next_iter_dir=$base_dir/$stage_dir/iter${iter_plus_one}

mkdir -p $next_iter_dir
cd $next_iter_dir

#ln -s $iter_dir/STATIONS.REF_ENU.subset.new STATIONS.REF_ENU.subset
#ln -s $iter_dir/STATIONS.REF_ENU.subset.include STATIONS.REF_ENU.common
#awk '$1!~/#/{printf "%s.%s\n",$1,$2}' STATIONS.REF_ENU.subset > grep.f
#grep -f grep.f $data_dir/data/event.txt > $next_iter_dir/event.txt

cp -a $iter_dir/misfit_par $next_iter_dir/

cp -a $iter_dir/control_file $next_iter_dir/
sed -i "s/^iter_num=[0-9][0-9]*/iter_num=$iter_plus_one/" $next_iter_dir/control_file

ln -s $sem_utils_dir $next_iter_dir/utils

echo "Create new model!!"