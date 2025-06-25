# This script store chern number with dynamics T 
# Exampel: srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_all_T.py --Lx $Lx --Ly $Ly --nshell $nshell --tf 4 --seed0 $seed --mu $mu --es 10 
# Exampel: srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_all_T.py --Lx 12 --Ly 12 --nshell 2 --tf 4 --seed0 0 --mu 1 --es 10 
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
    Lx,Ly, nshell,mu,sigma,tf,seed=inputs
    gtn2_torch=GTN2_torch(Lx=Lx,Ly=Ly,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)
    mu_list=[mu]
    tau_list=[(1,1),(1,-1)]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        for tau in tau_list:
            gtn2_torch.a_i[mu,tau],gtn2_torch.b_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=True,mu=mu,nkx=Lx,nky=Ly)
            gtn2_torch.A_i[mu,tau],gtn2_torch.B_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=False,mu=mu,nkx=Lx,nky=Ly)
    return gtn2_torch

def run(inputs):
    Lx,Ly, nshell,mu,sigma,tf,seed=inputs
    gtn2_torch=GTN2_torch(Lx=Lx,Ly=Ly,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)

    print( torch.sum((1-torch.diag(gtn2_torch.C_m,1)[::2])/2) / gtn2_torch.L )
    gtn2_torch.a_i = gtn2_dummy.a_i
    gtn2_torch.b_i = gtn2_dummy.b_i
    gtn2_torch.A_i = gtn2_dummy.A_i
    gtn2_torch.B_i = gtn2_dummy.B_i


    Chern_list =[]
    I2_list = []
    for i in tqdm(range(tf*gtn2_torch.Lx)):
        measure_feedback_layer(gtn2_torch,mu=mu)
        randomize(gtn2_torch,measure=True)
        if sigma>0:
            randomize_inter(gtn2_torch,scale=sigma)
        Chern_list.append(gtn2_torch.chern_number_quick(selfaverage=True))
        I2_list.append(gtn2_torch.bipartite_mutual_information_quasi_1d(selfaverage=True,partition=4))
        
    return {'nu':Chern_list, 'I2': I2_list}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx','-Lx',type=int)
    parser.add_argument('--Ly','-Ly',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--tf','-tf',type=int,default=1,help='the final time will be tf*L')
    parser.add_argument('--sigma','-sigma',type=float,default=0,help='local U(1) scale,  phase is [0,2pi*sigma]')

    args=parser.parse_args()

    st=time.time()
    inputs=[(args.Lx, args.Ly, args.nshell, args.mu,args.sigma, args.tf, seed+args.seed0) for seed in range(args.es)]
    nu_list=[]
    I2_list=[]
    gtn2_dummy=dummy(inputs[0])
    for inp in inputs:
        rs= run(inp)
        nu_list.append(rs['nu'])
        I2_list.append(rs['I2'])
    
    fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_tf{args.tf}_all_T.pt'
    torch.save({
        'Chern':torch.tensor(nu_list),
        'I2':torch.tensor(I2_list),
        'args':args},fn)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


