#!/bin/bash

#SBATCH --job-name=stratexp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=50GB
#SBATCH --time=10:00:00
#SBATCH --output=stratexpepn0-%A.out
#SBATCH --mail-type=end
#SBATCH --mail-user=hs3673@nyu.edu

echo 'Running'

module purge
module load python/intel/3.8.6

cd /home/hs3673/storage/hs3673/RLAttacks/attacks/
for id in 5 6 7 8 9
do
  for i in 1 2
  do
for env in "CartpoleEnvironment" "MountainCarEnvironment"
do
for method in "FQE" "WDR" "IS"
do
 if [[ $method = "IS" ]]
          then
            for is_type in "is" "pdis" "cpdis"
            do
              python experiments/fqe_experiments.py --method_type $method --is_type $is_type --env $env --experiment_id $i --dataset_id $id &
            done
            else
              python experiments/fqe_experiments.py --method_type $method --env $env --experiment_id $i --dataset_id $id &
        fi
done
done
done
done