#!/bin/bash

# setup mesh folders, generate the batch script to run SEM meshfem3D

proc_per_node=20

#====== command line args
run_dir=${1:?[arg]need run_dir(for all output)}
par_dir=${2:?[arg]need par_dir(for Par_file,STATIONS,*SOLUTION)}
mesh_dir=${3:?[arg]need mesh_dir(for DATABASES/*)}
sem_dir=${4:?[arg]need sem_dir(for code, DATA/*)}
mpiexec=${5:?[arg]need mpiexec (e.g. ibrun or mpirun)}

if [ -d "$run_dir" ]
then
    echo "[WARN] run_dir($run_dir) exists, delete!"
    rm -rf $run_dir
fi
mkdir $run_dir

if [ ! -d "$par_dir" ]
then
    echo "[ERROR] par_dir($par_dir) does NOT exist!"
    exit 1
elif [ ! -f "$par_dir/Par_file" ]
then
    echo "[WARN] $par_dir/Par_file does NOT exist!"
elif [ ! -f "$par_dir/CMTSOLUTION" ] && [ ! -f "$par_dir/FORCESOLUTION" ]
then
    echo "[WARN] $par_dir/CMTSOLUTION does NOT exist!"
elif [ ! -f "$par_dir/STATIONS" ]
then
    echo "[WARN] $par_dir/STATIONS does NOT exist!"
fi

if [ ! -d "$mesh_dir" ]
then
    echo "[ERROR] mesh_dir($mesh_dir) does NOT exit!"
    exit 1
fi

run_dir=$(readlink -f $run_dir)
par_dir=$(readlink -f $par_dir)
mesh_dir=$(readlink -f $mesh_dir)
sem_dir=$(readlink -f $sem_dir)

#====== setup run_dir
cd $run_dir
mkdir DATA DATABASES_MPI OUTPUT_FILES

# link data files: topography, bathymetry, etc.
cd $run_dir/DATA

#cp -a $par_dir/* .
cp -L $par_dir/Par_file .
cp -L $par_dir/*SOLUTION .
cp -L $par_dir/STATIONS .

# check DT: time step
DT=$(grep "Maximum suggested time step" $mesh_dir/OUTPUT_FILES/output_generate_databases.txt |\
 sed "s/.*=//" | awk '{printf "%f",$1}')

echo
echo "Maximum suggested time step: $DT"
echo "time step set in Par_file is: " $(grep ^DT Par_file)
echo

#sed -i "/^[\s]*SAVE_MESH_FILES/s/=.*/= .false./" Par_file
#sed -i "/^[\s]*MODEL/s/=.*/= GLL/" Par_file

# backup Par_file into OUTPUT_FILES/
cp -L Par_file CMTSOLUTION FORCESOLUTION STATIONS $run_dir/OUTPUT_FILES/

# link mesh database
cd $run_dir/DATABASES_MPI
ln -s $mesh_dir/DATABASES_MPI/proc*_external_mesh.bin .
if [ -f $mesh_dir/DATABASES_MPI/proc000000_attenuation.bin ]
then
  ln -s $mesh_dir/DATABASES_MPI/proc*_attenuation.bin .
fi

#ln -s $mesh_dir/DATABASES_MPI/proc*_Database .

# OUTPUT_FILES
cp $mesh_dir/OUTPUT_FILES/values_from_mesher.h $run_dir/OUTPUT_FILES
#cp $mesh_dir/OUTPUT_FILES/addressing.txt $run_dir/OUTPUT_FILES
#cp $mesh_dir/OUTPUT_FILES/addressing.txt $run_dir/DATABASES_MPI

# generate sbatch job file
nproc=$(grep ^NPROC $par_dir/Par_file | awk '{print $NF}')
nnode=$(echo "$nproc $proc_per_node" | awk '{a=$1/$2}END{print (a==int(a))?a:int(a)+1}')
 
cat <<EOF > $run_dir/syn.job
#!/bin/bash
#SBATCH -J syn
#SBATCH -o $run_dir/syn.job.o%j
#SBATCH -N $nnode
#SBATCH -n $nproc
#SBATCH -t 01:00:00
##SBATCH -p normal
##SBATCH --mail-user=kai.tao@utexas.edu
##SBATCH --mail-type=begin
##SBATCH --mail-type=end

cd $run_dir
${mpiexec} -np $nproc $sem_dir/bin/xspecfem3D

# move seismograms into the diretory seis/
cd $run_dir/OUTPUT_FILES
mkdir seis
mv *.sem? seis/

EOF
#END
