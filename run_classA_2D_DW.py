# This script is used to compute the domain wall dynamics of the 2D class A system
from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time
from utils_torch import *



def measure_feedback_layer_dw_line(gtn2,overlap,geometry,truncate=False):
    ilist = range(gtn2.Lx)
    jlist = range(gtn2.Ly)
    ij_list = [(i,j) for i in (ilist) for j in (jlist)]
    margin=0 if overlap else gtn2.nshell
    inner_list = range(gtn2.Lx//5+margin,gtn2.Lx//5*4-margin)
    outer_list = range(gtn2.Lx//5-margin,gtn2.Lx//5*4+margin)
    if geometry == 'square':
        inner_region=set([(i,j) for i in inner_list for j in inner_list])
        outer_region=set([(i,j) for i in ilist for j in jlist if i not in outer_list or j not in outer_list])
    elif geometry == 'strip':
        inner_region=set([(i,j) for i in inner_list for j in jlist])
        outer_region=set([(i,j) for i in ilist for j in jlist if i not in outer_list])
    region_inner=inner_region if truncate else None
    region_outer=outer_region if truncate else None
    # for i,j in tqdm(ij_list,desc='measure with feedback'):
    for i,j in ij_list:
        if (i,j) in inner_region:
            gtn2.measure_feedback(ij = [i,j],mu=1,region=region_inner,tau=(1,1))
            gtn2.measure_feedback(ij = [i,j],mu=1,region=region_inner,tau=(1,-1))
        elif (i,j) in outer_region:
            gtn2.measure_feedback(ij = [i,j],mu=3,region=region_outer,tau=(1,1))
            gtn2.measure_feedback(ij = [i,j],mu=3,region=region_outer,tau=(1,-1))



def randomize(gtn2,measure=True):
    # for i in tqdm(range(2*gtn2.L+1,4*gtn2.L,2),desc='randomize'):
    for i in range(2*gtn2.L+1,4*gtn2.L,2):
        # print([i, (i+1)%(2*gtn2.L)+2*gtn2.L])
        gtn2.randomize([i, (i+1)%(2*gtn2.L)+2*gtn2.L])
    if measure:
        # for i in tqdm(range(2*gtn2.L,4*gtn2.L,2),desc='measure'):
        for i in range(2*gtn2.L,4*gtn2.L,2):
            gtn2.measure_single_mode_Born([i,i+1],mode=[1])

def dummy(inputs):
    L,nshell,tf,truncate,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)
    mu_list=[1,3]
    tau_list=[(1,1),(1,-1)]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        for tau in tau_list:
            gtn2_torch.a_i[mu,tau],gtn2_torch.b_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=True,mu=mu,nkx=5000,nky=5000)
            gtn2_torch.A_i[mu,tau],gtn2_torch.B_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=False,mu=mu,nkx=5000,nky=5000)
    return gtn2_torch

def run(inputs):
    L,nshell,tf,truncate,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=2,)
    mu_list=[1,3]
    tau_list = [(1,1),(1,-1)]

    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        for tau in tau_list:
            gtn2_torch.a_i[mu,tau],gtn2_torch.b_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=True,mu=mu,nkx=5000,nky=5000)
            gtn2_torch.A_i[mu,tau],gtn2_torch.B_i[mu,tau] = amplitude_fft_nshell_gpu(gtn2_torch.nshell,gtn2_torch.device,tau=tau,geometry='square',lower=False,mu=mu,nkx=5000,nky=5000)
    
    ilist=np.arange(0,gtn2_torch.Lx)
    jlist=np.arange(0,gtn2_torch.Ly)
    subregion_m = torch.hstack((
        torch.from_numpy(gtn2_torch.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).cuda(),
        torch.from_numpy(gtn2_torch.linearize_idx_span(ilist = ilist,jlist=jlist,layer=1)).cuda())
    )
    EC_list=[]
    C_r_list=[]

    EC_list.append( gtn2_torch.entanglement_contour(subregion_m,fermion_idx=False,Gamma=gtn2_torch.C_m).reshape((2,ilist.shape[0],jlist.shape[0],2,2)).sum(axis=(-1,-2))) 
    C_r_list.append( gtn2_torch.local_Chern_marker(gtn2_torch.C_m,))

    for i in tqdm(range(tf*gtn2_torch.Lx)):
        measure_feedback_layer_dw_line(gtn2_torch,overlap=True,geometry='strip',truncate=truncate)
        randomize(gtn2_torch,measure=True)
        
        EC_list.append( gtn2_torch.entanglement_contour(subregion_m,fermion_idx=False,Gamma=gtn2_torch.C_m).reshape((2,ilist.shape[0],jlist.shape[0],2,2)).sum(axis=(-1,-2))) 
        C_r_list.append( gtn2_torch.local_Chern_marker(gtn2_torch.C_m,))
    
    return EC_list,C_r_list

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int,default=2)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--tf','-tf',type=int,default=20)
    parser.add_argument('--truncate','-truncate',action='store_true')
    args=parser.parse_args()

    st=time.time()
    inputs=[(args.L, args.nshell, args.tf, args.truncate,seed+args.seed0) for seed in range(args.es)]
    EC_list = []
    C_r_list = []
    gtn2_dummy=dummy(inputs[0])
    for inp in inputs:
        EC, C_r = run(inp)
        EC_list.append(EC)
        C_r_list.append(C_r)

    fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_es{args.es}_seed{args.seed0}_tf{args.tf}{"_truncate" if args.truncate else ""}.pt'
    torch.save({'EC':EC_list,'C_r':C_r_list,'args':args},fn)
    print('Time elapsed: {:.4f}'.format(time.time()-st))



