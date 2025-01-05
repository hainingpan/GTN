import torch
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
    if torch.abs(torch.einsum(Gamma,[0,1],Gamma,[1,0],[0])+1).max() > err:
        Gamma=purify(Gamma)
        Gamma=Gamma-Gamma.T
        Gamma/=2


def purify(A):
    U, _, Vh=torch.linalg.svd(A)
    return U@Vh

# def purify(A):
#     # purify A, see App. B2 in PhysRevB.106.134206
#     val,vec=torch.linalg.eigh(A/1j)
#     mask=val<0
#     val[mask]=-1
#     val[~mask]=1
#     val=val+0j
#     return -(vec@torch.diag(val)@vec.conj().T).imag

def get_O(rng,n,device,dtype):
    # rng=np.random.default_rng(rng)
    A=torch.normal(0,1,size=(n,n),generator=rng,device=device,dtype=dtype)/2
    A=A-A.T
    return torch.linalg.matrix_exp(A)

def chern_number_quick(Gamma,A_idx,B_idx,C_idx,device,dtype,U1=True,):
    P=(torch.eye(Gamma.shape[0],device=device,dtype=dtype)-1j*Gamma)/2
    # P_AB=P[np.ix_(A_idx,B_idx)]
    P_AB=P[A_idx[:,None],B_idx[None,:]]
    # P_BC=P[np.ix_(B_idx,C_idx)]
    P_BC=P[B_idx[:,None],C_idx[None,:]]
    # P_CA=P[np.ix_(C_idx,A_idx)]
    P_CA=P[C_idx[:,None],A_idx[None,:]]
    # P_AC=P[np.ix_(A_idx,C_idx)]
    P_AC=P[A_idx[:,None],C_idx[None,:]]
    # P_CB=P[np.ix_(C_idx,B_idx)]
    P_CB=P[C_idx[:,None],B_idx[None,:]]
    # P_BA=P[np.ix_(B_idx,A_idx)]
    P_BA=P[B_idx[:,None],A_idx[None,:]]
    h=12*torch.pi*1j*(torch.einsum("jk,kl,lj->jkl",P_AB,P_BC,P_CA)-torch.einsum("jl,lk,kj->jkl",P_AC,P_CB,P_BA))
    # assert np.abs(h.imag).max()<1e-10, "Imaginary part of h is too large"
    nu=h.real.sum()
    # return h
    if U1:
        return nu/2
    else:
        return nu

