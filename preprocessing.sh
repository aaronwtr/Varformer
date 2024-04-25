#/bin/bash
#$ -cwd        
#$ -l h_rt=03:30:0   
#$ -l h_vmem=11G   
#$ -m ea
#$ -N train_gdtp
#$ -e error.log
#$ -o output.log
#$ -pe smp 8

export OMP_NUM_THREADS=1
module load anaconda3
conda activate gdtp
python src/main.py
