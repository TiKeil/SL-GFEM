#!/bin/bash
 
#SBATCH --nodes=4                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5000
#SBATCH --partition=express          # on which partition to submit the job
#SBATCH --time=2:00:00             # the max wallclock time (time limit your job will run)
 
#SBATCH --job-name=test_pumslod_exp2     # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=t_keil02@uni-muenster.de # your mail address

# output file
#SBATCH --output /scratch/tmp/t_keil02/pumslod/experiment_2.dat

# load modules
module add palma/2019a
module add GCC/8.2.0-2.31.1
module add OpenMPI/3.1.3
module add Python/3.7.2
module add SuiteSparse/5.4.0-METIS-5.1.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMPI_MCA_mpi_warn_on_fork=0

. ~/PUMSLOD/venv/bin/activate

sleep 1

echo "Launching job:"
srun python -u /home/t/t_keil02/PUMSLOD/scripts/main_experiment_2.py /home/t/t_keil02/PUMSLOD/scripts/ /scratch/tmp/t_keil02/pumslod/mpi_storage/exp2

if [ $? -eq 0 ]
then
    echo "Job ${SLURM_JOB_ID} completed successfully!"
else
    echo "FAILURE: Job ${SLURM_JOB_ID}"
fi
