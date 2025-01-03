import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as sla
import numpy.linalg as nla

def zeros_like(A):
    return sp.lil_matrix(A.shape,dtype=A.dtype)
def sparse_ix_set(A,ix1,ix2,value):
    value_coo=value.tocoo()
    return sp.coo_matrix((value_coo.data,(np.array(ix1)[value_coo.row],np.array(ix2)[value_coo.col])),shape=A.shape).tocsr()

def sparse_ix_bar_ix_set(ix,ix_bar,value_bb,value_b,value,shape):
    """fill the matrix as 
    A=np.zeros(shape)
    A[np.ix_(ix_bar,ix_bar)]=value_bb
    A[np.ix_(ix,ix_bar)]=value_b
    A[np.ix_(ix_bar,ix)]=-value_b.T
    A[np.ix_(ix,ix)]=value
    """
    value_bb=value_bb.tocoo()
    value_b=value_b.tocoo()
    value_bT=-value_b.T
    value=value.tocoo()
    ix=np.array(ix)
    ix_bar=np.array(ix_bar)
    return sp.coo_matrix(
    (np.hstack([value_bb.data,value_b.data,value_bT.data,value.data]),
    (np.hstack([ix_bar[value_bb.row],ix[value_b.row],ix_bar[value_bT.row],ix[value.row]]),
    np.hstack([ix_bar[value_bb.col],ix_bar[value_b.col],ix[value_bT.col],ix[value.col]]))),
    shape=shape
    ).tocsr()
    # return sp.coo_matrix((value_coo.data,(np.array(ix1)[value_coo.row],np.array(ix2)[value_coo.col])),shape=A.shape).tocsr()

def P_contraction_sp(Gamma,Upsilon,ix,ix_bar,Gamma_like=None,reset_Gamma_like=True):
    """ same analytical expression for contraction as _contraction(), differences:
    1. assume intput and output tensor have the same shape, thus, it should be Gamma(L,R) -- Upsilon (L,R), where Gamma_R = Upsilon_L =Upsilon_R, such that in-place operator is applied here.
    2. manually compute the inverse of mat2 before
    Here, Gamma is m, and proj is Upsilon
    Assume Upsilon = [A,B;C,D], the logic is to first compute C= (1+ Gamma_RR @ Upsilon_LL)^{-1}, (where then B=-C.T) 
    then A= Upsilon_LL @C, D= Gamma_RR@ C.T
    ---
    reset_Gamma_like: in usual case, because each application of the gate will be like the brick layer, therefore, ix_bar will overwrite the previous. However, for the staircase pattern, one should reset it. 
    """
    Gamma_RR=Gamma[np.ix_(ix,ix)]
    Gamma_LR=Gamma[np.ix_(ix_bar,ix)]
    Gamma_LL=Gamma[np.ix_(ix_bar,ix_bar)]
    Upsilon_LL=Upsilon[:len(ix),:len(ix)]
    Upsilon_RR=Upsilon[len(ix):,len(ix):]
    Upsilon_RL=Upsilon[len(ix):,:len(ix)]
    eye=sp.eye(len(ix))
    try: 
        core=(Gamma_RR@Upsilon_LL+eye).todense()
        C=sp.csr_matrix(nla.inv(core))
    except:
        raise ValueError("the contraction will lead to a vanishing state")
    A=Upsilon_LL@C
    D=Gamma_RR@C.T
    tmp_bb=Gamma_LR@A@Gamma_LR.T + Gamma_LL
    tmp_b=Upsilon_RL@C@Gamma_LR.T
    tmp=Upsilon_RR+Upsilon_RL@D@Upsilon_RL.T
    return sparse_ix_bar_ix_set(ix,ix_bar,tmp_bb,tmp_b,tmp,Gamma.shape).eliminate_zeros()




    # Psi_LL=sparse_ix_set(Gamma,ix_bar,ix_bar,tmp_bb )
    # Psi_RL=sparse_ix_set(Gamma,ix,ix_bar,tmp_b)
    # Psi_RR=sparse_ix_set(Gamma,ix,ix,tmp)
    # Psi_LR=-Psi_RL.T
    # return Psi_LL+Psi_LR+Psi_RL+Psi_RR