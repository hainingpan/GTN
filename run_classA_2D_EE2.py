from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time
from utils_torch import *


def measure_feedback_layer(gtn2,):
    margin_x=0 
    ilist = range(margin_x,gtn2.Lx-margin_x)
    margin_y=0
    jlist = range(margin_y,gtn2.Ly-margin_y)
    ij_list = [(i,j) for i in (ilist) for j in (jlist)]
    for i,j in (ij_list):
        gtn2.measure_feedback(ij = [i,j])

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

def run(inputs):
    L, nshell,mu,sigma,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)

    mu_list=[mu]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        gtn2_torch.a_i[mu],gtn2_torch.b_i[mu] = amplitude(gtn2_torch.nshell,tau=[0,1],geometry='square',lower=True,mu=mu,C=1)
        gtn2_torch.A_i[mu],gtn2_torch.B_i[mu] = amplitude(gtn2_torch.nshell,tau=[1,0],geometry='square',lower=False,mu=mu,C=1)
    

    A_idx_0,B_idx_0,C_idx_0 = gtn2_torch.generate_tripartite_circle()

    for i in tqdm(range(gtn2_torch.Lx)):
        measure_feedback_layer(gtn2_torch)
        randomize(gtn2_torch,measure=True)
        if sigma>0:
            randomize_inter(gtn2_torch,scale=sigma)
        
    st=time.time()
    EE = gtn2_torch.half_cut_entanglement_entropy(selfaverage=True)
    # nu=gtn2_torch.chern_number_quick(selfaverage=True)
    # print('Chern number calculated in {:.4f}'.format(time.time()-st))
    # TMI=gtn2_torch.tripartite_mutual_information(selfaverage=True)
    print('EE calculated in {:.4f}'.format(time.time()-st))
    # free, total = torch.cuda.mem_get_info()
    # mem_used_MB = (total - free) / 1024 ** 2
    # print(mem_used_MB)
    return EE


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--sigma','-sigma',type=float,default=0,help='local U(1) scale,  phase is [0,2pi*sigma]')

    args=parser.parse_args()

    st=time.time()
    inputs=[(args.L, args.nshell, args.mu,args.sigma, seed+args.seed0) for seed in range(args.es)]
    EE_list=[]
    for inp in inputs:
        EE = run(inp)
        EE_list.append(EE)

    
    fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_EE.pt'
    torch.save({'EE':torch.tensor(EE_list),'args':args},fn)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


