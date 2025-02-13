# This script store the order parameter with dynamics T 
from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time
from utils_torch import *


def measure_feedback_layer(gtn2,mu):
    margin_x=0 
    ilist = range(margin_x,gtn2.Lx-margin_x)
    margin_y=0
    jlist = range(margin_y,gtn2.Ly-margin_y)
    ij_list = [(i,j) for i in (ilist) for j in (jlist)]
    for i,j in (ij_list):
        # gtn2.measure_feedback(ij = [i,j])
        gtn2.measure_feedback(ij = [i,j],tau=(1,1),mu=mu)
        gtn2.measure_feedback(ij = [i,j],tau=(1,-1),mu=mu)

def randomize(gtn2,measure=True):
    # for i in tqdm(range(2*gtn2.L+1,4*gtn2.L,2),desc='randomize'):
    for i in range(2*gtn2.L+1,4*gtn2.L,2):
        # print([i, (i+1)%(2*gtn2.L)+2*gtn2.L])
        gtn2.randomize([i, (i+1)%(2*gtn2.L)+2*gtn2.L])
    if measure:
        # for i in (range(2*gtn2.L,4*gtn2.L,2),desc='measure'):
        for i in (range(2*gtn2.L,4*gtn2.L,2)):
            gtn2.measure_single_mode_Born([i,i+1],mode=[1])

def randomize_inter(gtn2,scale=1):
    # for i in tqdm(range(2*gtn2.L+1,4*gtn2.L,2),desc='randomize'):
    for i in range(0,2*gtn2.L,2):
        gtn2.randomize([i, (i+1)%(2*gtn2.L)],scale=scale)

def dummy(inputs):
    L, nshell,mu,sigma,tf,seed=inputs
    # nshell=(L-1)//2
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)
    mu_list=[mu]
    tau_list=[(1,1),(1,-1)]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        for tau in tau_list:
            gtn2_torch.a_i[mu,tau],gtn2_torch.b_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=True,mu=mu,nkx=L,nky=L)
            gtn2_torch.A_i[mu,tau],gtn2_torch.B_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=False,mu=mu,nkx=L,nky=L)
    return gtn2_torch

def run(inputs):
    L, nshell,mu,sigma,tf,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=True,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)

    
    gtn2_torch.a_i = gtn2_dummy.a_i
    gtn2_torch.b_i = gtn2_dummy.b_i
    gtn2_torch.A_i = gtn2_dummy.A_i
    gtn2_torch.B_i = gtn2_dummy.B_i


    A_idx_0,B_idx_0,C_idx_0 = gtn2_torch.generate_tripartite_circle()
    OP_list =[]
    EE_j_list = []
    for i in tqdm(range(tf*gtn2_torch.Lx)):
        measure_feedback_layer(gtn2_torch,mu=mu)
        randomize(gtn2_torch,measure=True)
        if sigma>0:
            randomize_inter(gtn2_torch,scale=sigma)
        # gtn2_dummy.C_m = gtn2_torch.C_m
        # OP_list.append(gtn2_dummy.order_parameter(mu=mu,tau_list = [(1,1),(1,-1)]))
        OP_list.append(gtn2_torch.order_parameter(mu=mu,tau_list = [(1,1),(1,-1)]))
        # EE_j_list.append(gtn2_torch.half_cut_entanglement_y_entropy(selfaverage=True))
        
    # return OP_list, EE_j_list
    return OP_list


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--tf','-tf',type=int,default=1,help='the final time will be tf*L')
    parser.add_argument('--sigma','-sigma',type=float,default=0,help='local U(1) scale,  phase is [0,2pi*sigma]')

    args=parser.parse_args()

    st=time.time()
    inputs=[(args.L, args.nshell, args.mu,args.sigma, args.tf, seed+args.seed0) for seed in range(args.es)]
    OP_list=[]
    EE_j_list=[]
    gtn2_dummy=dummy(inputs[0])
    for inp in inputs:
        # OP, EE_j = run(inp)
        OP= run(inp)
        OP_list.append(OP)
        # EE_j_list.append(EE_j)

    
    # fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_tf{args.tf}_T.pt'
    # fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_tf{args.tf}_full_T.pt'
    fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_tf{args.tf}_discrete_T.pt'
    torch.save({
        'OP':torch.tensor(OP_list),
        # 'EE_j':torch.tensor(EE_j_list),
        'args':args},fn)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


