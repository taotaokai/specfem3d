#!/bin/bash

# copy files from previous iter_dir

source_dir=${1:?[arg]need previous iter_dir}
target_dir=${2:?[arg]need current iter_dir}

source_dir=$(readlink -f $source_dir)

mkdir $target_dir

#cp -a $source_dir/old .
#ln -s $source_dir/STATIONS.REF_ENU.subset.new STATIONS.REF_ENU.subset
#awk '$1!~/#/{printf "%s.%s\n",$1,$2}' STATIONS.REF_ENU.subset > grep.f
#grep -f grep.f $data_dir/data/event.txt > ${outfile}

cp -a $source_dir/STATIONS* $target_dir/
cp -a $source_dir/misfit_par $target_dir/
cp -a $source_dir/control_file $target_dir/

find $source_dir -maxdepth 1 -type l | xargs -I@ cp -a @ $target_dir/

#rm $target_dir/model