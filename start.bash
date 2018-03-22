#!/bin/bash
work=/gss_gpfs_scratch/kadioglu.b/gaussian90
cd ${work}
mkdir reg
mkdir rob


# There are 9 different rate values
for rate in "1" "3" "5" "10" "15" "20" "25" "30" "40"
do
    cd ${work}
    cd reg
    mkdir $rate
    cd $rate
    sbatch ../../reg.bash $rate
done


for rate in "1" "3" "5" "10" "15" "20" "25" "30" "40"
do
    cd ${work}
    cd rob
    mkdir $rate
    cd $rate
    sbatch ../../rob.bash $rate
done