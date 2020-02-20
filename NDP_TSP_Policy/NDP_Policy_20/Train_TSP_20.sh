#!/bin/bash

n=5
while [ $n -le 20 ]
do
    echo "$n"
    python DNN_train_N_P.py --N_node $n
    n=$(($n+1))
done
python FT_train_N_TSP.py --N_node 21


