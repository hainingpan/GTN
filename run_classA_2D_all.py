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
        # gtn2.measure_feedback(ij = [i,j],tau=(1,0),mu=mu)
        # gtn2.measure_feedback(ij = [i,j],tau=(0,1),mu=mu)
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

def run(inputs):
    Lx,Ly, nshell,mu,sigma,seed=inputs
    gtn2_torch=GTN2_torch(Lx=Lx,Ly=Ly,history=False,random_init=False,random_U1=True,bcx=1,bcy=1,seed=seed,orbit=2,nshell=nshell,layer=2,replica=1,complex128=True)

    print( torch.sum((1-torch.diag(gtn2_torch.C_m,1)[::2])/2) / gtn2_torch.L )
    gtn2_torch.a_i = gtn2_dummy.a_i
    gtn2_torch.b_i = gtn2_dummy.b_i
    gtn2_torch.A_i = gtn2_dummy.A_i
    gtn2_torch.B_i = gtn2_dummy.B_i


    for i in tqdm(range(gtn2_torch.Lx)):
        measure_feedback_layer(gtn2_torch,mu=mu)
        randomize(gtn2_torch,measure=True)
        if sigma>0:
            randomize_inter(gtn2_torch,scale=sigma)
        
    st=time.time()
    # EE_i = gtn2_torch.half_cut_entanglement_x_entropy(selfaverage=True)
    # EE_j = gtn2_torch.half_cut_entanglement_y_entropy(selfaverage=True)
    # print('EE calculated in {:.4f}'.format(time.time()-st))
    nu=gtn2_torch.chern_number_quick(selfaverage=True)
    print('Chern number calculated in {:.4f}'.format(time.time()-st))
    # TMI=gtn2_torch.tripartite_mutual_information(selfaverage=True)
    TMI=gtn2_torch.tripartite_mutual_information_quasi_1d(selfaverage=True)
    print('TMI calculated in {:.4f}'.format(time.time()-st))
    C_m=gtn2_torch.C_m_selfaverage(n=1)
    C_m2=gtn2_torch.C_m_selfaverage(n=2)
    # print('Average C_m calculated in {:.4f}'.format(time.time()-st))
    # SA = torch.tensor([gtn2_torch.entanglement_y_entropy(ly=ly,selfaverage=True) for ly in range(1,gtn2_torch.Ly//2+1)])
    # print('SA calculated in {:.4f}'.format(time.time()-st))
    # OP =gtn2_torch.order_parameter(mu=mu,tau_list = [(1,1),(1,-1)])
    # print('OP calculated in {:.4f}'.format(time.time()-st))
    I2=gtn2_torch.bipartite_mutual_information_quasi_1d(selfaverage=True,partition=4)
    print('I2 calculated in {:.4f}'.format(time.time()-st))
    # return EE_i,EE_j,nu,TMI,C_m,C_m2, SA, OP
    # return nu,TMI,I2,C_m
    return {'nu':nu,'TMI':TMI,'I2':I2,'C_m':C_m,'C_m2':C_m2}
    # return I2



def dummy(inputs):
    Lx,Ly, nshell,mu,sigma,seed=inputs
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

def correlation_length(C_m,replica,layer,Lx,Ly,):
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))
    Cr_i=torch.stack([C_ij[i,j,(i+torch.arange(Lx//2)+1)%Lx,j] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    Cr_j=torch.stack([C_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    c_ij=C_m.reshape(D+D)[0,1,:,:,:,:,0,1,:,:,:,:].mean(dim=(2,3,6,7))
    cr_i=torch.stack([c_ij[i,j,(i+torch.arange(Lx//2)+1)%Lx,j] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    cr_j=torch.stack([c_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    return Cr_i,Cr_j,cr_i,cr_j

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--Lx','-Lx',type=int)
    parser.add_argument('--Ly','-Ly',type=int)
    parser.add_argument('--nshell','-nshell',type=int)
    parser.add_argument('--mu','-mu',type=float)
    parser.add_argument('--es','-es',type=int,default=10)
    parser.add_argument('--seed0','-seed0',type=int,default=0)
    parser.add_argument('--sigma','-sigma',type=float,default=0,help='local U(1) scale,  phase is [0,2pi*sigma]')

    args=parser.parse_args()

    st=time.time()
    inputs=[(args.Lx, args.Ly,args.nshell, args.mu,args.sigma, seed+args.seed0) for seed in range(args.es)]
    EE_i_list=[]
    EE_j_list=[]
    TMI_list=[]
    nu_list=[]
    SA_list=[]
    OP_list=[]
    I2_list=[]
    gtn2_dummy=dummy(inputs[0])
    gtn2_dummy.C_m.zero_()
    C_m_sq=gtn2_dummy.C_m.clone()
    for inp in inputs:
        # EE_i,EE_j,nu,TMI,C_m,C_m2, SA, OP = run(inp)
        rs = run(inp)
        # nu,TMI,I2,C_m = rs['nu'],rs['TMI'],rs['I2'],rs['C_m']
        # I2 = run(inp)

        # EE_i_list.append(EE_i)
        # EE_j_list.append(EE_j)
        nu_list.append(rs['nu'])
        TMI_list.append(rs['TMI'])
        gtn2_dummy.C_m+= rs['C_m']
        C_m_sq+=rs['C_m2']
        # SA_list.append(SA)
        # OP_list.append(OP)
        I2_list.append(rs['I2'])

    gtn2_dummy.C_m/=args.es
    eigvals=torch.linalg.eigvalsh(gtn2_dummy.C_m/1j)
    eigvals_t=torch.linalg.eigvalsh(gtn2_dummy.C_m[:2*gtn2_dummy.L,:2*gtn2_dummy.L]/1j)
    eigvals_b=torch.linalg.eigvalsh(gtn2_dummy.C_m[2*gtn2_dummy.L:,2*gtn2_dummy.L:]/1j)
    nu_ave_unreg=gtn2_dummy.chern_number_quick(selfaverage=True)
    gtn2_dummy.C_m[:2*gtn2_dummy.L,:2*gtn2_dummy.L] = purify(gtn2_dummy.C_m[:2*gtn2_dummy.L,:2*gtn2_dummy.L])
    # gtn2_dummy.C_m = purify(gtn2_dummy.C_m)
    nu_ave=gtn2_dummy.chern_number_quick(selfaverage=True)
    sq_ave_C_m = gtn2_dummy.C_m**2
    sq_ave_Cr_i,sq_ave_Cr_j, sq_ave_cr_i,sq_ave_cr_j =correlation_length(sq_ave_C_m,replica=1,layer=2,Lx=args.Lx,Ly=args.Ly)

    ave_sq_C_m = C_m_sq/args.es
    ave_sq_Cr_i,ave_sq_Cr_j, ave_sq_cr_i,ave_sq_cr_j =correlation_length(ave_sq_C_m,replica=1,layer=2,Lx=args.Lx,Ly=args.Ly)

    
    # fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_all.pt'
    # fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_all_pub.pt'
    fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_all_pub_uni.pt'
    # fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_all_I2.pt'
    # fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_all_I2_8.pt'
    torch.save({
        # 'EE_i':torch.tensor(EE_i_list),
        # 'EE_j':torch.tensor(EE_j_list),
        'TMI':torch.tensor(TMI_list),
        'Chern':torch.tensor(nu_list),
        'Chern_ave_unregularized': nu_ave_unreg,
        'Chern_ave':nu_ave,
        'sq_ave_Cr_i':sq_ave_Cr_i,
        'sq_ave_Cr_j':sq_ave_Cr_j,
        'sq_ave_cr_i':sq_ave_cr_i,
        'sq_ave_cr_j':sq_ave_cr_j,
        'ave_sq_Cr_i':ave_sq_Cr_i,
        'ave_sq_Cr_j':ave_sq_Cr_j,
        'ave_sq_cr_i':ave_sq_cr_i,
        'ave_sq_cr_j':ave_sq_cr_j,
        'eigvals':eigvals,
        'eigvals_t':eigvals_t,
        'eigvals_b':eigvals_b,
        # 'SA':torch.stack(SA_list),
        # 'OP':torch.tensor(OP_list),
        'I2':torch.tensor(I2_list),
        # 'I2_8':torch.tensor(I2_list),
        'args':args
        },fn)

    # fn=f'class_A_2D_Lx{args.Lx}_Ly{args.Ly}_nshell{args.nshell}_mu{args.mu:.2f}_sigma{args.sigma:.3f}_es{args.es}_seed{args.seed0}_EE_fix.pt'
    # torch.save({'EE_i':torch.tensor(EE_i_list),'EE_j':torch.tensor(EE_j_list),'args':args},fn)
    
    print('Time elapsed: {:.4f}'.format(time.time()-st))


