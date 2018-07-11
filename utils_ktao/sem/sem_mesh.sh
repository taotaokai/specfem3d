#!/bin/bash

# setup mesh folders, generate the batch script to run SEM meshfem3D

proc_per_node=20

#====== command line args
run_dir=${1:?[arg]need run_dir(for all output)}
par_dir=${2:?[arg]need par_dir(for Par_file,STATIONS,CMTSOLUTION)}
sem_dir=${3:?[arg]need sem_dir(for code, DATA/*)}
mpiexec=${4:?[arg]need mpiexec (e.g. ibrun or mpirun)}

if [ -d "$run_dir" ]
then
    echo "[WARN] run_dir($run_dir) exists, delete!"
    rm -rf $run_dir
fi
mkdir -p $run_dir

if [ ! -d "$par_dir" ]
then
    echo "[ERROR] par_dir($par_dir) does NOT exist!"
    exit 1
elif [ ! -f "$par_dir/Par_file" ]
then
    echo "[ERROR] $par_dir/Par_file does NOT exist!"
    exit 1
elif [ ! -f "$par_dir/CMTSOLUTION" ] && [ ! -f "$par_dir/FORCESOLUTION" ]
then
    echo "[ERROR] $par_dir/CMTSOLUTION or FORCESOLUTION does NOT exist!"
    exit 1
fi

if [ ! -d "$sem_dir" ]
then
    echo "[ERROR] sem_dir($sem_dir) does NOT exit!"
    exit 1
fi

run_dir=$(readlink -f $run_dir)
par_dir=$(readlink -f $par_dir)
sem_dir=$(readlink -f $sem_dir)

#====== setup run_dir
cd $run_dir
mkdir DATA DATABASES_MPI OUTPUT_FILES

cd $run_dir/DATA

cp -rL $par_dir/* .

#sed -i "/^[\s]*SAVE_MESH_FILES/s/=.*/= .false./" Par_file
#sed -i "/^[\s]*MODEL/s/=.*/= GLL/" Par_file

# backup Par_file into OUTPUT_FILES/
cp -L Par_file CMTSOLUTION FORCESOLUTION $run_dir/OUTPUT_FILES/

# generate sbatch job file
nproc=$(grep ^NPROC $par_dir/Par_file | awk '{print $NF}')
nnode=$(echo "$nproc $proc_per_node" | awk '{a=$1/$2}END{print (a==int(a))?a:int(a)+1}')
 
cat <<EOF > $run_dir/mesh.job
#!/bin/bash
#SBATCH -J mesh
#SBATCH -o $run_dir/mesh.job.o%j
#SBATCH -N $nnode
#SBATCH -n $nproc
#SBATCH -t 00:10:00
##SBATCH -p normal
##SBATCH --mail-user=kai.tao@utexas.edu
##SBATCH --mail-type=begin
##SBATCH --mail-type=end

cd $run_dir

${mpiexec} -np $nproc $sem_dir/bin/xmeshfem3D

${mpiexec} -np $nproc $sem_dir/bin/xgenerate_databases

EOF
#END
