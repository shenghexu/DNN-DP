import os
import time
import argparse
import importlib
import scipy.io
import numpy as np
import numba as nb
import math


@nb.jit(nopython=True, parallel=True, nogil=True)
def dis_mtx_from_axis(C, posits):
    N_node =  posits.shape[1]
    for i in nb.prange(posits.shape[0]):
        for x in range(N_node):
            for y in range(N_node):
                C[i,x,y] = math.sqrt( (posits[i,x,0] - posits[i,y,0])**2 + (posits[i,x,1] - posits[i,y,1])**2 )
            C[i,x,x]=0.0
        #np.fill_diagonal(C[i],0)

    return C


def Gen_C_mtx(Total_num, N_node):
    posits =  np.random.uniform(low=0.0, high=1.0, size=(Total_num, N_node, 2))
    C = np.zeros((Total_num, N_node, N_node))
    # for i in range(Total_num):
    # 	for x in range(N_node):
    # 		for y in range(N_node):
    # 			C[i,x,y] = np.sqrt( (posits[i,x,0] - posits[i,y,0])**2 + (posits[i,x,1] - posits[i,y,1])**2 )
    # 	np.fill_diagonal(C[i],0)
    C = dis_mtx_from_axis(C, posits)

    return C

def Gen_C_mtx_FT(Total_num, N_node):
    posits =  np.random.uniform(low=0.0, high=1.0, size=(Total_num, N_node, 2))
    posits[:,0,:] = posits[:,N_node-1,:]
    C = np.zeros((Total_num, N_node, N_node))

    C = dis_mtx_from_axis(C, posits)

    return C
