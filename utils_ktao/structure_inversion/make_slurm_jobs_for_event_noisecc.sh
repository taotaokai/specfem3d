#!/bin/bash
# Make jobs files for slurm 
# structure inversion

control_file=${1:?[arg]need control_file}
event_id=${2:?[arg]need event_id}

# source control_file
source $control_file

## link the required directories to your iter_dir
#sem_build_dir=$iter_dir/specfem3d_globe # bin/x...
mesh_dir=$iter_dir/mesh # DATABASES_MPI/proc*_reg1_solver_data.bin
#mesh_perturb_dir=$iter_dir/mesh_perturb # DATABASES_MPI/proc*_reg1_solver_data.bin
##model_dir=$iter_dir/mesh/DATABASES_MPI # proc*_reg1_vph,vpv,vsv,vsh,eta,rho.bin
##!!! model files should reside in mesh_dir/DATABASES_MPI
#data_dir=$iter_dir/events # <event_id>/data,dis
#utils_dir=$sem_utils_dir/structure_inversion # sem_utils/utils/misfit_v0
 
#====== define variables
# directories
event_dir=$iter_dir/$event_id
misfit_dir=$event_dir/misfit
figure_dir=$misfit_dir/figure
slurm_dir=$event_dir/slurm
# job scripts for slurm
mkdir -p $slurm_dir
syn_job=$slurm_dir/syn.job
misfit_job=$slurm_dir/misfit.job
kernel_job=$slurm_dir/kernel.job
#hess_job=$slurm_dir/hess.job
#precond_job=$slurm_dir/precond.job
perturb_job=$slurm_dir/perturb.job
search_job=$slurm_dir/search.job
#hess_diag_job=$slurm_dir/hess_diag.job
#hess_model_product_job=$slurm_dir/hess_model_product.job
#hess_kernel_job=$slurm_dir/hess_kernel.job
# hessian-random model product
hess_syn_job=$slurm_dir/hess_syn.job
hess_misfit_job=$slurm_dir/hess_misfit.job
hess_kernel_job=$slurm_dir/hess_kernel.job

# database file
#mkdir -p $misfit_dir
db_file=$misfit_dir/misfit.pkl
# station file
station_file=$event_dir/DATA/STATIONS.lla
if [ ! -f "$station_file" ]
then
  echo "[ERROR] $station_file does NOT exist!"
  exit -1
fi
# cmt file
cmt_file=$event_dir/DATA/FORCESOLUTION.lla
if [ ! -f "$cmt_file" ]
then
  echo "[ERROR] $cmt_file does NOT exist!"
  exit -1
fi
# misfit par file
misfit_par=$event_dir/DATA/misfit_par.py
if [ ! -f "$misfit_par" ]
then
  echo "[ERROR] $misfit_par does NOT exist!"
  exit -1
fi

#====== syn: forward simulation
cat <<EOF > $syn_job
#!/bin/bash
#SBATCH -J ${event_id}.syn
#SBATCH -o $syn_job.o%j
#SBATCH -N $slurm_nnode
#SBATCH -n $slurm_nproc
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_forward

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

out_dir=output_syn

mkdir -p $event_dir/DATA
cd $event_dir/DATA

# convert STATION to REF_ENU
$python_exec $sem_utils_dir/meshfem3d/convert_STATIONS_from_geodesic_to_REF_ENU.py $sem_config_dir/meshfem3D_files/mesh_par.py  $sem_config_dir/meshfem3D_files/topo.grd STATIONS.lla tmp
grep -v nan tmp > STATIONS
rm tmp

# convert FORCESOLUTION to REF_ENU
$python_exec $sem_utils_dir/meshfem3d/convert_FORCESOLUTION_from_geodesic_to_REF_ENU.py $sem_config_dir/meshfem3D_files/mesh_par.py  $sem_config_dir/meshfem3D_files/topo.grd FORCESOLUTION.lla FORCESOLUTION

cp $mesh_dir/DATA/Par_file .
sed -i "/^SIMULATION_TYPE/s/=.*/= 1/" Par_file
sed -i "/^SAVE_FORWARD/s/=.*/= .true./" Par_file
sed -i "/^SAVE_MESH_FILES/s/=.*/= .false./" Par_file

rm -rf $event_dir/DATABASES_MPI
mkdir $event_dir/DATABASES_MPI
ln -s $mesh_dir/DATABASES_MPI/proc*_external_mesh.bin $event_dir/DATABASES_MPI
ln -s $mesh_dir/DATABASES_MPI/proc*_attenuation.bin $event_dir/DATABASES_MPI

cd $event_dir
rm -rf \$out_dir OUTPUT_FILES
mkdir \$out_dir
ln -sf \$out_dir OUTPUT_FILES

#cp $mesh_dir/OUTPUT_FILES/addressing.txt OUTPUT_FILES
cp $mesh_dir/OUTPUT_FILES/values_from_mesher.h OUTPUT_FILES
cp -L DATA/Par_file OUTPUT_FILES
cp -L DATA/STATIONS OUTPUT_FILES
cp -L DATA/FORCESOLUTION OUTPUT_FILES

${slurm_mpiexec} $sem_build_dir/bin/xspecfem3D

mkdir $event_dir/\$out_dir/seis
mv $event_dir/\$out_dir/*.sem? $event_dir/\$out_dir/seis

mkdir $event_dir/\$out_dir/sac
$python_exec $sem_utils_dir/meshfem3d/convert_ascii_in_REF_ENU_to_sac_in_local_ENZ.py $sem_config_dir/meshfem3D_files/mesh_par.py  $event_dir/DATA/FORCESOLUTION.lla $event_dir/DATA/STATIONS.lla $sem_band_code $event_dir/\$out_dir/seis $event_dir/\$out_dir/sac

chmod a+w -R $event_dir/save_forward
rm -rf $event_dir/save_forward
mv $event_dir/DATABASES_MPI $event_dir/save_forward
chmod a-w -R $event_dir/save_forward

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

EOF

#====== misfit
cat <<EOF > $misfit_job
#!/bin/bash
#SBATCH -J ${event_id}.misfit
#SBATCH -o $misfit_job.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_misfit

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

cd $event_dir

rm -rf $misfit_dir
mkdir -p $misfit_dir
$python_exec $sem_utils_dir/misfit/read_data.py \
  $misfit_par \
  $db_file \
  $event_dir/DATA/CMTSOLUTION.lla \
  $data_dir/$event_id/data/channel.txt \
  $event_dir/output_syn/sac \
  $data_dir/$event_id/dis

$python_exec $sem_utils_dir/misfit/measure_misfit.py $misfit_par $db_file

$python_exec $sem_utils_dir/misfit/output_misfit.py $db_file $misfit_dir/misfit.txt

rm -rf $figure_dir
mkdir -p $figure_dir
$python_exec $sem_utils_dir/misfit/plot_misfit.py $misfit_par $db_file $figure_dir

#------ adjoint source for kernel simulation
rm -rf $event_dir/adj_kernel
mkdir -p $event_dir/adj_kernel/local_ENU
$python_exec $sem_utils_dir/misfit/output_adj.py $misfit_par $db_file $event_dir/adj_kernel/local_ENU

# coordinate conversion (local ENZ -> REF_ENU)
$python_exec $sem_utils_dir/meshfem3d/convert_ascii_from_local_ENZ_to_REF_ENU.py \
  $sem_config_dir/meshfem3D_files/mesh_par.py \
  $event_dir/DATA/FORCESOLUTION.lla \
  $event_dir/DATA/STATIONS.lla \
  $adj_band_code \
  $event_dir/adj_kernel/local_ENU \
  $event_dir/adj_kernel/

# make STATIONS_ADJOINT
cd $event_dir/adj_kernel
ls *Z.adj | sed 's/..Z\.adj$//' |\
  awk -F"." '{printf "%s[ ]*%s.%s[ ]\n",\$1,\$2,\$3}' > grep_pattern
grep -f $event_dir/adj_kernel/grep_pattern $event_dir/DATA/STATIONS \
  > $event_dir/adj_kernel/STATIONS_ADJOINT

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

EOF

#====== kernel simulation
cat <<EOF > $kernel_job
#!/bin/bash
#SBATCH -J ${event_id}.kernel
#SBATCH -o $kernel_job.o%j
#SBATCH -N $slurm_nnode
#SBATCH -n $slurm_nproc
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_adjoint

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

out_dir=output_kernel

cd $event_dir/DATA
chmod u+w Par_file
sed -i "/^SIMULATION_TYPE/s/=.*/= 3/" Par_file
sed -i "/^SAVE_FORWARD/s/=.*/= .false./" Par_file
sed -i "/^ANISOTROPIC_KL/s/=.*/= .false./" Par_file
sed -i "/^SAVE_TRANSVERSE_KL_ONLY/s/=.*/= .false./" Par_file
sed -i "/^APPROXIMATE_HESS_KL/s/=.*/= .false./" Par_file

cp -f $event_dir/adj_kernel/STATIONS_ADJOINT $event_dir/DATA/
rm -rf $event_dir/SEM
ln -s $event_dir/adj_kernel $event_dir/SEM

cd $event_dir

rm -rf \$out_dir OUTPUT_FILES
mkdir \$out_dir
ln -sf \$out_dir OUTPUT_FILES

rm -rf $event_dir/DATABASES_MPI
mkdir $event_dir/DATABASES_MPI

ln -s $event_dir/save_forward/*.bin $event_dir/DATABASES_MPI

cp $mesh_dir/OUTPUT_FILES/values_from_mesher.h OUTPUT_FILES
#cp $mesh_dir/OUTPUT_FILES/addressing.txt OUTPUT_FILES
cp -L DATA/Par_file OUTPUT_FILES
cp -L DATA/STATIONS_ADJOINT OUTPUT_FILES
#cp -L DATA/CMTSOLUTION OUTPUT_FILES

cd $event_dir
${slurm_mpiexec} $sem_build_dir/bin/xspecfem3D

#mkdir $event_dir/\$out_dir/sac
#mv $event_dir/\$out_dir/*.sac $event_dir/\$out_dir/sac

mkdir $event_dir/\$out_dir/kernel
mv $event_dir/DATABASES_MPI/*_kernel.bin $event_dir/\$out_dir/kernel/

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo
EOF

#====== perturb: forward simulation of perturbed model
cat <<EOF > $perturb_job
#!/bin/bash
#SBATCH -J ${event_id}.perturb
#SBATCH -o $perturb_job.o%j
#SBATCH -N $slurm_nnode
#SBATCH -n $slurm_nproc
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_forward
#SBATCH --mail-user=kai.tao@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

cd $event_dir/DATA
sed -i "/^SIMULATION_TYPE/s/=.*/= 1/" Par_file
sed -i "/^SAVE_FORWARD/s/=.*/= .false./" Par_file
 
#for dmodel in dvp dvsv dvsh
for dmodel in perturb
do

  out_dir=output_\${dmodel}
 
  rm -rf $event_dir/DATABASES_MPI
  mkdir $event_dir/DATABASES_MPI
  ln -s $iter_dir/mesh_\${dmodel}/DATABASES_MPI/*.bin $event_dir/DATABASES_MPI
  
  cd $event_dir

  rm -rf \$out_dir OUTPUT_FILES
  mkdir \$out_dir
  ln -sf \$out_dir OUTPUT_FILES
  
  cp $iter_dir/mesh_\${dmodel}/OUTPUT_FILES/addressing.txt OUTPUT_FILES
  cp -L DATA/Par_file OUTPUT_FILES
  cp -L DATA/STATIONS OUTPUT_FILES
  cp -L DATA/CMTSOLUTION OUTPUT_FILES
  
  ${slurm_mpiexec} $sem_build_dir/bin/xspecfem3D
  
  mkdir $event_dir/\$out_dir/sac
  mv $event_dir/\$out_dir/*.sac $event_dir/\$out_dir/sac

done

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo
EOF

#====== search: cc linearized seismograms for chosen step sizes
cat <<EOF > $search_job
#!/bin/bash
#SBATCH -J ${event_id}.search
#SBATCH -o $search_job.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_misfit
#SBATCH --mail-user=kai.tao@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

$utils_dir/waveform_der_dmodel.py $misfit_par $db_file $event_dir/output_perturb/sac model

$utils_dir/grid_search_dmodel.py $misfit_par $db_file $misfit_dir/grid_search_dmodel.txt

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

EOF

#///////////////////////// Hessian simulation

#====== hess_syn: forward simulation of randomly perturbed model
cat <<EOF > $hess_syn_job
#!/bin/bash
#SBATCH -J ${event_id}.hess_syn
#SBATCH -o $hess_syn_job.o%j
#SBATCH -N $slurm_nnode
#SBATCH -n $slurm_nproc
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_hess_forward
#SBATCH --mail-user=kai.tao@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

cd $event_dir/DATA
sed -i "/^SIMULATION_TYPE/s/=.*/= 1/" Par_file
sed -i "/^SAVE_FORWARD/s/=.*/= .true./" Par_file

for tag in ${hess_model_names}
do

  echo ====== \$tag
 
  out_dir=output_syn_\${tag}
  
  rm -rf $event_dir/DATABASES_MPI
  mkdir $event_dir/DATABASES_MPI
  ln -s $iter_dir/mesh_\${tag}/DATABASES_MPI/*.bin $event_dir/DATABASES_MPI
  
  cd $event_dir
  
  rm -rf \$out_dir OUTPUT_FILES
  mkdir \$out_dir
  ln -sf \$out_dir OUTPUT_FILES
  
  cp $iter_dir/mesh_\${tag}/OUTPUT_FILES/addressing.txt OUTPUT_FILES
  cp -L DATA/Par_file OUTPUT_FILES
  cp -L DATA/STATIONS OUTPUT_FILES
  cp -L DATA/CMTSOLUTION OUTPUT_FILES
  
  ${slurm_mpiexec} $sem_build_dir/bin/xspecfem3D
  
  mkdir $event_dir/\$out_dir/sac
  mv $event_dir/\$out_dir/*.sac $event_dir/\$out_dir/sac

done

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo
EOF

#====== hess_misfit
cat <<EOF > $hess_misfit_job
#!/bin/bash
#SBATCH -J ${event_id}.hess_misfit
#SBATCH -o $hess_misfit_job.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_hess_misfit
#SBATCH --mail-user=kai.tao@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

cd $event_dir

for tag in ${hess_model_names}
do

  echo ====== \$tag

  # make adjoint source
  rm -rf $event_dir/adj_kernel_\${tag}
  mkdir -p $event_dir/adj_kernel_\${tag}
  
  $utils_dir/output_adj_for_perturbed_waveform.py \
    $misfit_par \
    $db_file \
    $event_dir/output_syn_\${tag}/sac \
    $event_dir/adj_kernel_\${tag}
  
  # make STATIONS_ADJOINT
  cd $event_dir/adj_kernel_\${tag}
  ls *Z.adj | sed 's/..Z\.adj$//' |\
    awk -F"." '{printf "%s[ ]*%s.%s[ ]\n",\$1,\$2,\$3}' > grep_pattern
  grep -f $event_dir/adj_kernel_\${tag}/grep_pattern $event_dir/DATA/STATIONS \
    > $event_dir/adj_kernel_\${tag}/STATIONS_ADJOINT

done

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

EOF

#====== kernel simulation for randomly perturbed model
cat <<EOF > $hess_kernel_job
#!/bin/bash
#SBATCH -J ${event_id}.hess_kernel
#SBATCH -o $hess_kernel_job.o%j
#SBATCH -N $slurm_nnode
#SBATCH -n $slurm_nproc
#SBATCH -p $slurm_partition
#SBATCH -t $slurm_timelimit_hess_adjoint
#SBATCH --mail-user=kai.tao@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

echo
echo "Start: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo

for tag in ${hess_model_names}
do

  echo ====== \${tag}

  out_dir=output_kernel_\${tag}
  
  cd $event_dir/DATA
  chmod u+w Par_file
  sed -i "/^SIMULATION_TYPE/s/=.*/= 3/" Par_file
  sed -i "/^SAVE_FORWARD/s/=.*/= .false./" Par_file
  sed -i "/^ANISOTROPIC_KL/s/=.*/= .true./" Par_file
  sed -i "/^SAVE_TRANSVERSE_KL_ONLY/s/=.*/= .false./" Par_file
  sed -i "/^APPROXIMATE_HESS_KL/s/=.*/= .false./" Par_file
  
  cp -f $event_dir/adj_kernel_\${tag}/STATIONS_ADJOINT $event_dir/DATA/
  rm -rf $event_dir/SEM
  ln -s $event_dir/adj_kernel_\${tag} $event_dir/SEM
  
  cd $event_dir
  
  rm -rf \$out_dir OUTPUT_FILES
  mkdir \$out_dir
  ln -sf \$out_dir OUTPUT_FILES
  
  cp $mesh_dir/OUTPUT_FILES/addressing.txt OUTPUT_FILES
  cp -L DATA/Par_file OUTPUT_FILES
  cp -L DATA/STATIONS_ADJOINT OUTPUT_FILES
  cp -L DATA/CMTSOLUTION OUTPUT_FILES
  
  cd $event_dir
  ${slurm_mpiexec} $sem_build_dir/bin/xspecfem3D
  
  mkdir $event_dir/\$out_dir/sac
  mv $event_dir/\$out_dir/*.sac $event_dir/\$out_dir/sac
  
  mkdir $event_dir/\$out_dir/kernel
  mv $event_dir/DATABASES_MPI/*reg1_cijkl_kernel.bin $event_dir/\$out_dir/kernel/
  mv $event_dir/DATABASES_MPI/*reg1_rho_kernel.bin $event_dir/\$out_dir/kernel/

done

echo
echo "Done: JOB_ID=\${SLURM_JOB_ID} [\$(date)]"
echo
EOF
