#!/bin/bash
work=/gss_gpfs_scratch/kadioglu.b/jaw_l5_a1


for seed in "1" #"2" "3" "4" "5" #"6" "7" "8" "9" "10"
do

    cd ${work}
    mkdir $seed
    cd $seed
    mkdir reg
    mkdir rob

    for rate in ".0001" ".001" ".006" ".012" ".020"
    do
        cd ${work}
        cd ${seed}
        cd reg
        mkdir $rate
        cd $rate
        sbatch ../../../reg.bash $rate 5 1 $seed
    done

    for rate in ".0001" ".001" ".006" ".012" ".020"
    do
        cd ${work}
        cd ${seed}
        cd rob
        mkdir $rate
        cd $rate
        sbatch ../../../rob.bash $rate 5 1 $seed
    done
done
