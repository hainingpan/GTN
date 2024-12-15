from GTN2 import *
from utils import *
import time
import argparse
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import numpy as np

def run(inputs):
    L, nshell, mu= inputs
    gtn2=GTN2(Lx=L,Ly=L,history=False,random_init=True,bcx=1,bcy=1,seed=0,orbit=2,nshell=nshell,op=True)
    gtn2.a_i,gtn2.b_i = amplitude(gtn2.nshell,tau=[0,1],geometry='square',lower=True,mu=mu)
    gtn2.A_i,gtn2.B_i = amplitude(gtn2.nshell,tau=[1,0],geometry='square',lower=False,mu=mu)

    def measure_feedback_layer():
        ij_list = [(i,j) for i in range(gtn2.Lx) for j in range(gtn2.Ly)]
        for i,j in tqdm(ij_list):
            gtn2.measure_feedback(ij = [i,j])
    def randomize():
        for i in tqdm(range(2*gtn2.L+1,4*gtn2.L,2)):
            # print([i, (i+1)%(4*gtn2.L)])
            gtn2.randomize([i, (i+1)%(4*gtn2.L)])

    nu_list =[]
    A_idx_0,B_idx_0,C_idx_0 = gtn2.generate_tripartite_circle()
    nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
    measure_feedback_layer()
    nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
    for i in range(20):
        randomize()
        measure_feedback_layer()
        nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
    return gtn2,nu_list

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    args=parser.parse_args()

    st=time.time()
    gtn2,nu_list = run([args.L,args.nshell,args.mu])
    
    with open(f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}.pickle','wb') as f:
        pickle.dump([gtn2,nu_list],f)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


