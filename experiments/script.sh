#!/bin/bash
 for env in "MountainCarEnvironment" "CancerEnvironment"
 do
 for seed in {1..10}
 do
   for method in "FQE" "IS" "WDR"
        if [[ $method = "IS" ]]
          then
            for is_type in "is" "pdis" "cpdis"
            do
              python experiments/fqe_experiments.py --method $method --seed $seed --is_type $is_type --env $env &
            done
            else
              python experiments/fqe_experiments.py --method $method --seed $seed --env $env
        fi
 done
 done
 echo "done"