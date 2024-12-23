from GTN2 import *
from utils import *
import time
import argparse
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor

import numpy as np
def measure_feedback_layer_monitor(gtn2,margin=0):
    ij_list = [(i,j) for i in range(margin,gtn2.Lx-margin) for j in range(margin,gtn2.Ly-margin)]
    for i,j in (ij_list):
        gtn2.measure_feedback(ij = [i,j])
        # nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
        # EE_list.append( gtn2.von_Neumann_entropy_m(subregion_m,fermion_idx=False))


    
def randomize_monitor(gtn2):
    for i in (range(2*gtn2.L+1,4*gtn2.L,2)):
        # print([i, (i+1)%(4*gtn2.L)])
        gtn2.randomize([i, (i+1)%(4*gtn2.L)])
        # nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
        # EE_list.append( gtn2.von_Neumann_entropy_m(subregion_m,fermion_idx=False))

def run(inputs):
    L, nshell, mu, seed, tf= inputs
    gtn2=GTN2(Lx=L,Ly=L,history=False,random_init=False,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,op=True)
    set_1= np.sort(gtn2.rng.choice(np.arange(0,4*gtn2.L,2),size=gtn2.L,replace=False))
    set_m1 = [i for i in np.arange(0,4*gtn2.L,2) if i not in set_1]
    gtn2.set(
        ij_list = [(i,i+1) for i in set_1], 
        n = [1 for i in set_1],
    )
    gtn2.set(
        ij_list = [(i,i+1) for i in set_m1], 
        n = [-1 for i in set_m1],
    )
    gtn2.a_i,gtn2.b_i = amplitude(gtn2.nshell,tau=[0,1],geometry='square',lower=True,mu=mu)
    gtn2.A_i,gtn2.B_i = amplitude(gtn2.nshell,tau=[1,0],geometry='square',lower=False,mu=mu)

    A_idx_0,B_idx_0,C_idx_0 = gtn2.generate_tripartite_circle()
    subregion_m =(gtn2.linearize_idx_span(ilist = np.arange(0,gtn2.Lx//2),jlist=np.arange(0,gtn2.Ly)))

    nu_list =[]
    EE_list = []
    nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
    EE_list.append( gtn2.von_Neumann_entropy_m(subregion_m,fermion_idx=False))

    measure_feedback_layer_monitor(gtn2)
    nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
    EE_list.append( gtn2.von_Neumann_entropy_m(subregion_m,fermion_idx=False))

    for i in range(tf):
        randomize_monitor(gtn2)
        measure_feedback_layer_monitor(gtn2)
        nu_list.append( chern_number_quick(gtn2.C_m,A_idx_0,B_idx_0,C_idx_0))
        EE_list.append( gtn2.von_Neumann_entropy_m(subregion_m,fermion_idx=False))
    return nu_list,EE_list


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--tf','-tf',type=int,default=10)
    args=parser.parse_args()

    st=time.time()
    inputs=[(args.L, args.nshell, args.mu, seed, args.tf) for seed in range(args.es)]
    with MPIPoolExecutor() as executor:
        rs=list(tqdm(executor.map(run,inputs),total=len(inputs)))
    # rs=list(tqdm(map(run,inputs),total=len(inputs)))

    rs=np.array(rs).reshape((args.es,2,args.tf+2))
    nu,EE=rs[:,0,:],rs[:,1,:]

    
    # gtn2,nu_list = run([args.L,args.nshell,args.mu])
    
    # with open(f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}.pickle','wb') as f:
    #     pickle.dump([gtn2,nu_list],f)
    with open(f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}_es{args.es}.pickle','wb') as f:
        pickle.dump([nu,EE],f)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


