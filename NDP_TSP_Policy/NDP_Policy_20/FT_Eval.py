import os
import time
import argparse
import importlib
import scipy.io
import numpy as np
from pdb import set_trace as bp
from utils import *

import torch
import torch.nn as nn


Train_Batch_size = 100
Batch_size = 10000
Test_num = 10000
learning_rate = 0.0001
batch_per_epoch = int(1000)


def Get_del_index_from_n(num):

    index_out = np.zeros((num-2, num-1), dtype=int)
    for i in range(num-2):
        temp_vec = np.delete(np.arange(num), i+1)
        temp_vec[0] = i+1
        index_out[i] = temp_vec

    return torch.LongTensor(index_out)



class DP_FT(object):
    def __init__(self, c_net_list, N_node):
        self.N_node = N_node
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net_list = c_net_list
        #self.device = device
        for i in range(len(self.net_list)):
            self.net_list[i] = self.net_list[i].to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net_list[self.N_node-4].parameters(), lr=learning_rate)

        self.optimizer_list = []
        self.optimizer_list.append(0) # Make sure we do not train DNN4

        for i in range(1, len(self.net_list)):
            self.optimizer_list.append(torch.optim.Adam(self.net_list[i].parameters(), lr=learning_rate))

        
        self.del_col_index_list = []
        for i in range(4, self.N_node):
            self.del_col_index_list.append(Get_del_index_from_n(i+1).to(self.device))

        self.last_del_col_index = Get_del_index_from_n(4).to(self.device)

        self.dummy_batch_idx = []
        for i in range(4 , self.N_node):
            self.dummy_batch_idx.append(self.Get_Train_data_dummy_idx(i+1))
        self.one_dummy_batch_idx = torch.arange(Train_Batch_size*batch_per_epoch).to(self.device)
        self.last_dummy_batch_idx = self.Get_Train_data_dummy_idx(4)


        self.Validate_1st_batch_idx = self.Get_Train_data_dummy_idx(self.N_node, 10000)
        self.Train_1st_batch_idx = self.Get_Train_data_dummy_idx(self.N_node, 100000)
        self.Validate_1st_del_col = Get_del_index_from_n(self.N_node).to(self.device)


    def Get_Train_data_dummy_idx(self, num, total_num=Train_Batch_size*batch_per_epoch):
        count = torch.arange(total_num).unsqueeze(1).repeat(1, num-1).to(self.device)
        return count



    

    def restore_all_models(self, ft_epoch):
        

        for i in range(self.N_node-3):
            if i == 0:
                file_name = './Models/DNN4_T_990.ckpt'
            else:
                file_name =  './FT_Models/DNN%d_T_%d.ckpt' %(i+4, ft_epoch)
                self.net_list[i].load_state_dict(torch.load(file_name))
                print('./FT_Models/DNN%d_T_%d.ckpt Restored' %(i+4, ft_epoch))
        for i in range(self.N_node-3):
            self.net_list[i] = self.net_list[i].to(self.device).eval()

    def batched_index_select(self, input, dim, index):
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)




    def Validate_data(self, data_C_mtx):

        Total_num = data_C_mtx.size()[0]
        
        mse_list = []

        
        NN_Rs = torch.zeros(Total_num).to(self.device)
        

        num_batch = int(Total_num/Batch_size)

        indices_temp = torch.zeros(Total_num, dtype=torch.long).to(self.device)

        for i in range(len(self.net_list)):
            self.net_list[i] = self.net_list[i].eval()

        for i in range(num_batch):
            b_st = i*Batch_size 
            b_ed = i*Batch_size + Batch_size
            batch_pred_temp, indices_temp[b_st:b_ed] = self.net_list[self.N_node-4](
                            data_C_mtx[b_st:b_ed].view( -1, self.N_node*self.N_node) )




        indices_temp = indices_temp.contiguous()

        NN_Rs = NN_Rs + data_C_mtx[self.one_dummy_batch_idx[0:Total_num], 0, indices_temp+1]


        c_dim3_idx_temp = torch.index_select(self.Validate_1st_del_col, 0, indices_temp)
        c_data_temp = data_C_mtx[self.Validate_1st_batch_idx, :, c_dim3_idx_temp]
        c_data_temp = c_data_temp[self.Validate_1st_batch_idx, :, c_dim3_idx_temp]

        c_data_temp = c_data_temp.contiguous()



        

        for jj in range(self.N_node-2, 2, -1):
            


            indices_temp = torch.zeros(Total_num, dtype=torch.long).to(self.device)

            for i in range(num_batch):
                b_st = i*Batch_size 
                b_ed = i*Batch_size + Batch_size

                __, indices_temp[b_st:b_ed] = self.net_list[jj-3](
                    c_data_temp[b_st:b_ed].view( -1, (jj+1)*(jj+1)))


            if jj == 3:

                NN_Rs = NN_Rs + c_data_temp[self.one_dummy_batch_idx[0:Total_num], 0, indices_temp+1]
                indices_temp = indices_temp.contiguous()
                c_dim3_idx_temp = torch.index_select(self.last_del_col_index, 0, indices_temp)
                c_data_temp_2 = c_data_temp[self.last_dummy_batch_idx[0:Total_num], :, c_dim3_idx_temp]
                c_data_temp = c_data_temp_2[self.last_dummy_batch_idx[0:Total_num], :, c_dim3_idx_temp].contiguous()


                NN_Rs =  NN_Rs + c_data_temp[:, 0, 1] + c_data_temp[:, 1, 2]
            else:

                NN_Rs = NN_Rs + c_data_temp[self.one_dummy_batch_idx[0:Total_num], 0, indices_temp+1]
                indices_temp = indices_temp.contiguous()
                c_dim3_idx_temp = torch.index_select(self.del_col_index_list[jj-4], 0, indices_temp)
                c_data_temp_2 = c_data_temp[self.dummy_batch_idx[jj-4][0:Total_num], :, c_dim3_idx_temp]
                c_data_temp = c_data_temp_2[self.dummy_batch_idx[jj-4][0:Total_num], :, c_dim3_idx_temp].contiguous()




                    

        return NN_Rs



  
    def test(self, test_Batch_size=Batch_size):
        
        if os.path.isfile('../TestSets/tsp20_test_seed1234.mat'):
            f_mat = scipy.io.loadmat('../TestSets/tsp20_test_seed1234.mat')
            posits = f_mat['test_data']
            posits_2 = np.zeros((10000,21,2))
            posits_2[:,0:20,:] = posits
            posits_2[:,20,:] = posits[:,0,:]
            Test_C_mtx = np.zeros((10000, self.N_node, self.N_node))
            Test_C_mtx  = dis_mtx_from_axis(Test_C_mtx , posits_2)
            f_mat = scipy.io.loadmat('../TestSets/tsp_Testset_G_%d.mat'%(self.N_node-1))
            Test_C_OptR = np.squeeze(f_mat['g_cost'])

            print('Loaded Testset')
        else:
            print('TestSet does not exist!!!!')


        Test_C_mtx_GPU = torch.from_numpy(Test_C_mtx).float().to(self.device)
        dummy_idx = torch.arange(test_Batch_size).to(self.device)
        start_time = time.time()
        fileName = 'DP_Eval_%d_train.txt' %(self.N_node)
        with open(fileName,'w') as filep:
            filep.write("Test time test_OptR test_R R_Relative\n")

        print('Start testing...')
        print('----------------------------')

       
        
        
        vali_OptR = np.mean(Test_C_OptR)



        for t in range(0, 100):

            start_time = time.time()

            NN_Rs_real = self.Validate_data(Test_C_mtx_GPU)

            t_end = time.time()

            print('Test [%8d] Time [%e] test_OptR [%e] test_R [%e] R_r [%.5f]' %
                (t, t_end - start_time, 
                      vali_OptR, NN_Rs_real.mean(), NN_Rs_real.mean()/vali_OptR))

            recordfile = open(fileName,'a')
            recordfile.write('%d %e %e %e %e \n' %
                (t, t_end - start_time, 
                      vali_OptR, NN_Rs_real.mean(), NN_Rs_real.mean()/vali_OptR))
            recordfile.close()

            



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--model', type=str, default='DNN_model_2')
    parser.add_argument('--prefix', type=str, default='DNN3')
    parser.add_argument('--N_node', type=int)
    parser.add_argument('--num_Epoch',  type=int, default=1500)
    parser.add_argument('--base_Epoch', type=int, default=10)
    parser.add_argument('--step_Epoch', type=int, default=50)

    args = parser.parse_args()
    net_list = []
    model_4 = importlib.import_module('DNN_model_4')
    net_list.append(model_4.DP_M())
    model = importlib.import_module('DNN_model_N')
    for i in range(5, args.N_node+1):   
        net_list.append(model.DP_M(i))
    sch = DP_FT(net_list, args.N_node)
    sch.restore_all_models(args.num_Epoch)
    sch.test(test_Batch_size=100)
