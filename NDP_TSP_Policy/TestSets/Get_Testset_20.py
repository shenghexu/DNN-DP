import argparse
import numpy as np
from gurobipy import *
import os
import time
import argparse
import importlib
from pdb import set_trace as bp
import scipy.io


def solve_sym_tsp(dis_mtx, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    n = len(dis_mtx)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of distance between each pair of points

    dist = {(i,j) :
        dis_mtx[i][j]
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
       vars[j,i] = vars[i,j] # edge in opposite direction

    


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))
    

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    #bp()
    assert len(tour) == n

    return m.objVal, tour


def solve_all_gurobi(N_node):
    mat_file_name = './Testset_%d.mat'%(N_node)
    f_mat = scipy.io.loadmat('../TestSets/Testset_%d.mat'%(N_node))
    Test_C_mtx = f_mat['Test_C_mtx']

    # For the pretraining datasets, we are finding the shortes path with given source and destination, no need to retrun. 
    
    

    t_start = time.time()
    # Add a city with 0 distance from the last city (ending point) and first city (starting point)
    # Then the optimum solution for TSP would always select this city as the last stop before returining.
    # Returning gives no extra cost compared with no returning. 
    dummy_col = 1000*np.ones((N_node,1))
    dummy_col[N_node-1] = 0
    dummy_col[0] = 0
    dummy_row = 1000*np.ones((1, N_node+1))
    dummy_row[0,0] = 0
    dummy_row[0,N_node-1] = 0
    g_sol = []
    for i in range(10000):
        if i%1000 ==0:
            print(i)
        dis_mtx_temp = np.squeeze(Test_C_mtx[i])
        dis_mtx_temp = np.concatenate((dis_mtx_temp, dummy_col), axis=1)
        dis_mtx_temp = np.concatenate((dis_mtx_temp, dummy_row), axis=0)
        cost_temp, __ = solve_sym_tsp(dis_mtx_temp.tolist())

        g_sol.append(cost_temp)

    print('G mean:%e'%(np.mean(g_sol)))
    g_cost = np.array(g_sol)

    t_end = time.time()
    print('Time spent:%.2f'%(t_end - t_start))
    scipy.io.savemat('../TestSets/Testset_G_%d.mat'%(N_node), dict(g_cost=g_cost))

def dis_mtx_from_axis(C, posits):
    N_node =  posits.shape[1]
    for i in range(posits.shape[0]):
        for x in range(N_node):
            for y in range(N_node):
                C[i,x,y] = math.sqrt( (posits[i,x,0] - posits[i,y,0])**2 + (posits[i,x,1] - posits[i,y,1])**2 )
            C[i,x,x]=0.0


    return C
    
def Generate(N_node):
    np.random.seed(1234)
    mat_file_name = './Testset_%d.mat'%(N_node)
    Total_num = 10000
    
    Test_posits = np.random.uniform(0.0, 1.0, size=(Total_num, N_node, 2))
    Test_C_mtx = np.zeros((Total_num, N_node, N_node))
    Test_C_mtx = dis_mtx_from_axis(Test_C_mtx, Test_posits)
    scipy.io.savemat(mat_file_name, dict(Test_posits=Test_posits, Test_C_mtx=Test_C_mtx))

if __name__ == '__main__':
    for i in range(4, 21):
        Generate(i)
        solve_all_gurobi(i)