Code for paper " Deep Neural Network Approximated Dynamic Programming for Combinatorial Optimization " (Accepted by AAAI-2020). 

## Running the code

### Dependencies
Python3.5  
numpy  
Pytorch1.1  
numba  
Scipy  
Gurobi

### Testing
Run 'FT_Eval.py' in the 'NDP_Policy_20' folder.
### Training
1. Run 'Get_Testset_20.py' in 'TestSets' folder to generate all the testsets.
2. Run script 'Train_TSP_20.sh'. By default the code trains for 1600 epochs of fine-tuning, which takes around 60 hours on a P40 with a CPU with two cores. 

### Cite our paper
```
@inproceedings{xu2020deep,
  title={Deep Neural Network Approximated Dynamic Programming for Combinatorial Optimization.},
  author={Xu, Shenghe and Panwar, Shivendra S and Kodialam, Murali S and Lakshman, TV},
  booktitle={AAAI},
  pages={1684--1691},
  year={2020}
}
```


