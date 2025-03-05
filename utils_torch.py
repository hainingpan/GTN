import torch
import time
def P_contraction_torch(Gamma,Upsilon,ix,ix_bar,device,err,Gamma_like=None,reset_Gamma_like=True,):
    """ same analytical expression for contraction as _contraction(), differences:
    1. assume intput and output tensor have the same shape, thus, it should be Gamma(L,R) -- Upsilon (L,R), where Gamma_R = Upsilon_L =Upsilon_R, such that in-place operator is applied here.
    2. manually compute the inverse of mat2 before
    Here, Gamma is m, and proj is Upsilon
    Assume Upsilon = [A,B;C,D], the logic is to first compute C= (1+ Gamma_RR @ Upsilon_LL)^{-1}, (where then B=-C.T) 
    then A= Upsilon_LL @C, D= Gamma_RR@ C.T
    ---
    reset_Gamma_like: in usual case, because each application of the gate will be like the brick layer, therefore, ix_bar will overwrite the previous. However, for the staircase pattern, one should reset it. 
    """
    Gamma_RR=Gamma[ix[:,None],ix[None,:]]
    Gamma_LR=Gamma[ix_bar[:,None],ix[None,:]]
    Upsilon_LL=Upsilon[:len(ix),:len(ix)]
    Upsilon_RR=Upsilon[len(ix):,len(ix):]
    Upsilon_RL=Upsilon[len(ix):,:len(ix)]
    eye=torch.eye(len(ix),device=device)
    try: 
        C=torch.linalg.inv(Gamma_RR@Upsilon_LL+eye)
        # lu,pivots=torch.linalg.lu_factor(Gamma_RR@Upsilon_LL+eye)
    except:
        raise ValueError("the contraction will lead to a vanishing state")
    A=Upsilon_LL@C
    D=Gamma_RR@C.T
    # A= torch.linalg.lu_solve(lu,pivots,Upsilon_LL,left=False)
    # D= torch.linalg.lu_solve(lu,pivots,Gamma_RR.T).T
    tmp=Gamma_LR@A@Gamma_LR.T
    if Gamma_like is None:
        Gamma_like=torch.zeros_like(Gamma)
    if reset_Gamma_like:
        Gamma_like.fill_(0)
    Gamma_like[ix_bar[:,None],ix_bar[None,:]]=tmp
    Gamma+=Gamma_like
    Gamma[ix[:,None],ix_bar[None,:]]=Upsilon_RL@C@Gamma_LR.T
    # Gamma[ix[:,None],ix_bar[None,:]]=torch.linalg.lu_solve(lu,pivots,Upsilon_RL,left=False)@Gamma_LR.T
    Gamma[ix[:,None],ix[None,:]]=Upsilon_RR+Upsilon_RL@D@Upsilon_RL.T
    Gamma[ix_bar[:,None],ix[None,:]]=-Gamma[ix[:,None],ix_bar[None,:]].T
    # why is it neccessary?
    # Gamma-=Gamma.T
    # Gamma/=2
    # print(torch.abs(torch.einsum(Gamma,[0,1],Gamma,[1,0],[0])+1).max())
    max_err=max_error(Gamma)
    if  max_err > err:
        st=time.time()
        print(f'Purification: {max_err}')
        Gamma=purify(Gamma)
        print('Purification done in {:.4f} {}'.format(time.time()-st,max_error(Gamma)))
        

def max_error(Gamma):
    return torch.abs(torch.einsum(Gamma,[0,1],Gamma,[1,0],[0])+1).max()
# def purify(A):
#     A = (A-A.T)/2
#     U, _, Vh=torch.linalg.svd(A)
#     return U@Vh

def purify(A):
    # purify A, see App. B2 in PhysRevB.106.134206
    val,vec=torch.linalg.eigh(A/1j)
    mask_neg=val<0
    mask_pos=val>0
    val[mask_neg]=-1
    val[mask_pos]=1
    val=val+0j
    A= -(vec@torch.diag(val)@vec.conj().T).imag
    A=A-A.T
    A/=2
    return A

def get_O(rng,n,device,dtype):
    # rng=np.random.default_rng(rng)
    A=torch.normal(0,1,size=(n,n),generator=rng,device=device,dtype=dtype)/2
    A=A-A.T
    return torch.linalg.matrix_exp(A)

def chern_number_quick(Gamma,A_idx,B_idx,C_idx,device,dtype,U1=True,):
    st=time.time()
    P=(torch.eye(Gamma.shape[0],device=device,dtype=dtype)-1j*Gamma)/2
    P_AB=P[A_idx[:,None],B_idx[None,:]]
    P_BC=P[B_idx[:,None],C_idx[None,:]]
    P_CA=P[C_idx[:,None],A_idx[None,:]]
    P_AC=P[A_idx[:,None],C_idx[None,:]]
    P_CB=P[C_idx[:,None],B_idx[None,:]]
    P_BA=P[B_idx[:,None],A_idx[None,:]]
    h=12*torch.pi*1j*(torch.einsum("jk,kl,lj->jkl",P_AB,P_BC,P_CA)-torch.einsum("jl,lk,kj->jkl",P_AC,P_CB,P_BA))
    # assert np.abs(h.imag).max()<1e-10, "Imaginary part of h is too large"
    nu=h.real.sum()
    print('Chern number done in {:.4f}'.format(time.time()-st))
    if U1:
        return nu/2
    else:
        return nu

def correlation_length(C_m,replica,layer,Lx,Ly,):
    """C_m is expected to be the element-wise square of the covariance matrix"""
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))
    Cr_i=torch.stack([C_ij[i,j,(i+torch.arange(Lx//2)+1)%Lx,j] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    Cr_j=torch.stack([C_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    c_ij=C_m.reshape(D+D)[0,1,:,:,:,:,0,1,:,:,:,:].mean(dim=(2,3,6,7))
    cr_i=torch.stack([c_ij[i,j,(i+torch.arange(Lx//2)+1)%Lx,j] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    cr_j=torch.stack([c_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in range(Lx) for j in range(Ly)]).mean(dim=0)
    return Cr_i,Cr_j,cr_i,cr_j

def correlation_i_length(C_m,replica,layer,Lx,Ly,i0):
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))
    Cr_j=torch.stack([C_ij[i,j,i,(j+torch.arange(Ly//2)+1)%Ly] for i in [i0] for j in range(Ly)]).mean(dim=0)
    return Cr_j

def correlation_i2_length(C_m,replica,layer,Lx,Ly,i0,i1):
    D=(replica,layer,Lx,Ly,2,2)
    C_ij=C_m.reshape(D+D)[0,0,:,:,:,:,0,0,:,:,:,:].mean(dim=(2,3,6,7))  # trace out the sublattice and Majorana degree of freedom
    Cr_j=torch.stack([C_ij[i0,j,i1,j] for j in range(Ly)]).mean()
    return Cr_j.item()


def fidelity(A,B):
    # https://doi.org/10.1063/1.5093326
    assert A.shape[0]==B.shape[0], f'A {A.shape[0]} has different dim than B{B.shape[0]}'
    L=A.shape[0]//2
    # identity=np.eye(A.shape[0])
    # id_AB=(identity-A@B)
    AB=-A@B
    AB.diagonal().add_(1.) # (I-A@B)
    AB.div_(2.)  # ../2
    prod1=torch.linalg.det(AB)
    if prod1 == 0:
        return 0.
    else:
        AB.mul_(2)
        AB_tilde = (A+B)@torch.linalg.inv(AB) # G_tilde=(A+B)@(1-A@B)^{-1}
        AB_tilde = AB_tilde@AB_tilde    # G_tilde^2
        AB_tilde.diagonal().add_(1.) # 1+..^2
        AB_tilde = sqrt(AB_tilde) # (..)^{1/2}
        AB_tilde.diagonal().add_(1.) # det(1+..)
        prod2=torch.linalg.det(AB_tilde)
        return torch.sqrt(prod1*prod2).item().real


def sqrt(A):
    val,vec=torch.linalg.eigh(A)
    val_sqrt = torch.sqrt(val+0j)
    vec=vec+0j
    return vec@torch.diag(val_sqrt)@vec.conj().T
