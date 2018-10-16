#!/bin/bash

control_file=${1:?[arg]need control_file}

#====== source control_file
if [ ! -f $control_file ] 
then
  echo "[ERROR] control_file does NOT exist!"
  exit -1
fi
source $control_file


#
mkdir $iter_dir/model

${python_exec} $sem_utils_dir/structure_inversion/sem_update_model_by_exp_dlnV.py \
  ${sem_nproc} $prev_iter_dir/model $model_tags \
  ${prev_iter_dir}/dmodel $dmodel_tags \
  ${iter_dir}/model $model_tags