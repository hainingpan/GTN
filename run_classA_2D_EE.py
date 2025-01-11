from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time
from utils_torch import *


def measure_feedback_layer(gtn2,):
    # margin_x=0 if gtn2.bcx==1 else gtn2.nshell
    margin_x=0 
    ilist = range(margin_x,gtn2.Lx-margin_x)
    # margin_y=0 if gtn2.bcy==1 else gtn2.nshell
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
    L, nshell,mu,seed=inputs
    gtn2_torch=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)

    mu_list=[mu]
    gtn2_torch.a_i={}
    gtn2_torch.b_i={}
    gtn2_torch.A_i={}
    gtn2_torch.B_i={}
    for mu in mu_list:
        gtn2_torch.a_i[mu],gtn2_torch.b_i[mu] = amplitude(gtn2_torch.nshell,tau=[0,1],geometry='square',lower=True,mu=mu,C=1)
        gtn2_torch.A_i[mu],gtn2_torch.B_i[mu] = amplitude(gtn2_torch.nshell,tau=[1,0],geometry='square',lower=False,mu=mu,C=1)
    
    # nu_list =[]
    # EE_list =[]
    # ilist=np.arange(0,gtn2_torch.Lx//2)
    # jlist=np.arange(0,gtn2_torch.Ly)
    # subregion_m = torch.hstack((
    #     torch.from_numpy(gtn2_torch.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).cuda(),
    #     torch.from_numpy(gtn2_torch.linearize_idx_span(ilist = ilist,jlist=jlist,layer=1)).cuda())
    # )

    A_idx_0,B_idx_0,C_idx_0 = gtn2_torch.generate_tripartite_circle()
    # nu_list.append(chern_number_quick(gtn2_torch.C_m,A_idx_0,B_idx_0,C_idx_0,device=gtn2_torch.device,dtype=gtn2_torch.dtype_float))
    # EE_list.append(gtn2_torch.von_Neumann_entropy_m(subregion_m,fermion_idx=False))

    for i in tqdm(range(gtn2_torch.Lx)):
        measure_feedback_layer(gtn2_torch)
        randomize(gtn2_torch,measure=True)
        # nu_list.append( chern_number_quick(gtn2_torch.C_m,A_idx_0,B_idx_0,C_idx_0,device=gtn2_torch.device,dtype=gtn2_torch.dtype_float))
        # EE_list.append(gtn2_torch.von_Neumann_entropy_m(subregion_m,fermion_idx=False))
    st=time.time()
    nu=gtn2_torch.chern_number_quick(selfaverage=True)
    print('Chern number calculated in {:.4f}'.format(time.time()-st))
    TMI=gtn2_torch.tripartite_mutual_information(selfaverage=True)
    print('TMI calculated in {:.4f}'.format(time.time()-st))
    free, total = torch.cuda.mem_get_info()
    mem_used_MB = (total - free) / 1024 ** 2
    print(mem_used_MB)
    return nu,TMI


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)

    args=parser.parse_args()

    st=time.time()
    inputs=[(args.L, args.nshell, args.mu, seed+args.seed0) for seed in range(args.es)]
    nu_list=[]
    TMI_list=[]
    for inp in inputs:
        nu, TMI = run(inp)
        nu_list.append(nu)
        TMI_list.append(TMI)

    
    fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_mu{args.mu:.2f}_es{args.es}_seed{args.seed0}_SE.pt'
    torch.save({'Chern':torch.tensor(nu_list),'TMI':torch.tensor(TMI_list),'args':args},fn)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


