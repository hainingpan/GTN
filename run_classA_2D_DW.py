from GTN2_torch import *
import torch
import argparse
from tqdm import tqdm
import time


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
    for i,j in ij_list:
        if (i,j) in inner_region:
            gtn2.measure_feedback(ij = [i,j],mu=1,region=region_inner)
        elif (i,j) in outer_region:
            gtn2.measure_feedback(ij = [i,j],mu=3,region=region_outer)


def randomize(gtn2,measure=True):
    for i in range(2*gtn2.L+1,4*gtn2.L,2):
        # print([i, (i+1)%(2*gtn2.L)+2*gtn2.L])
        gtn2.randomize([i, (i+1)%(2*gtn2.L)+2*gtn2.L])
    if measure:
        for i in range(2*gtn2.L,4*gtn2.L,2):
            gtn2.measure_single_mode_Born([i,i+1],mode=[1])

def run(inputs):
    L,nshell,tf,truncate=inputs
    gtn2=GTN2_torch(Lx=L,Ly=L,history=False,random_init=False,bcx=1,bcy=1,seed=0,orbit=2,nshell=nshell,layer=2,replica=2,)
    mu_list=[1,3]
    gtn2.a_i={}
    gtn2.b_i={}
    gtn2.A_i={}
    gtn2.B_i={}
    for mu in mu_list:
        gtn2.a_i[mu],gtn2.b_i[mu] = amplitude(gtn2.nshell,tau=[0,1],geometry='square',lower=True,mu=mu)
        gtn2.A_i[mu],gtn2.B_i[mu] = amplitude(gtn2.nshell,tau=[1,0],geometry='square',lower=False,mu=mu)
    
    ilist=np.arange(0,gtn2.Lx)
    jlist=np.arange(0,gtn2.Ly)
    subregion_m = torch.hstack((
        torch.from_numpy(gtn2.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).cuda(),
        torch.from_numpy(gtn2.linearize_idx_span(ilist = ilist,jlist=jlist,layer=1)).cuda())
    )
    EC_list=[]
    C_r_list=[]

    EC_list.append( gtn2.entanglement_contour(subregion_m,fermion_idx=False,Gamma=gtn2.C_m).reshape((2,ilist.shape[0],jlist.shape[0],2,2)).sum(axis=(-1,-2))) 
    C_r_list.append( gtn2.local_Chern_marker(gtn2.C_m,))

    for i in tqdm(range(tf)):
        measure_feedback_layer_dw_line(gtn2,overlap=True,geometry='strip',truncate=truncate)
        randomize(gtn2,measure=True)
        
        EC_list.append( gtn2.entanglement_contour(subregion_m,fermion_idx=False,Gamma=gtn2.C_m).reshape((2,ilist.shape[0],jlist.shape[0],2,2)).sum(axis=(-1,-2))) 
        C_r_list.append( gtn2.local_Chern_marker(gtn2.C_m,))
    
    return EC_list,C_r_list

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--nshell','-nshell',type=int,default=2)
    parser.add_argument('--tf','-tf',type=int,default=20)
    parser.add_argument('--truncate','-truncate',action='store_true')
    args=parser.parse_args()

    st=time.time()
    EC_list,C_r_list=run((args.L,args.nshell,args.tf,args.truncate))
    fn=f'class_A_2D_L{args.L}_nshell{args.nshell}_tf{args.tf}{"_truncate" if args.truncate else ""}.pt'
    torch.save({'EC':EC_list,'C_r':C_r_list},fn)
    print('Time elapsed: {:.4f}'.format(time.time()-st))



