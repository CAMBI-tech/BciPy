#!/bin/bash
work=/gss_gpfs_scratch/kadioglu.b/jaw_muscle50


for seed in "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
do

    cd ${work}
    mkdir reg
    mkdir rob

    for rate in ".0001" ".001" ".002" ".0035" ".005" ".0065" ".008" ".01" ".02" ".05"
    do
        cd ${work}
        cd reg
        mkdir $rate
        cd $rate
        sbatch ../../reg.bash $rate
    done

    for rate in ".0001" ".001" ".002" ".0035" ".005" ".0065" ".008" ".01" ".02" ".05"
    do
        cd ${work}
        cd rob
        mkdir $rate
        cd $rate
        sbatch ../../rob.bash $rate
    done
done