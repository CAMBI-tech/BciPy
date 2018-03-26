#!/bin/bash
work=/gss_gpfs_scratch/kadioglu.b/jaw_muscle50
cd ${work}
mkdir reg
mkdir rob


# There are 9 different rate values
for rate in ".0001" ".001" ".002" ".0035" ".005" ".0065" ".008" ".01" ".02" ".05"
do
    cd ${work}
    cd reg
    mkdir $rate
    cd $rate
    sbatch ../../reg.bash $rate
done


#for rate in ".01" ".03" ".05" ".1"
#do
#    cd ${work}
#    cd rob
#    mkdir $rate
#    cd $rate
#    sbatch ../../rob.bash $rate
#done