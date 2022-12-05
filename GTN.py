import numpy as np
import numpy.linalg as nla
from copy import copy

class GTN:
    def __init__(self,L,history=True,seed=None):
        self.L=L
        self.C_m=self.correlation_matrix()
        self.C_m_history=[self.C_m]
        self.history=history
        self.rng=np.random.default_rng(seed)
    
    def correlation_matrix(self):
        Omega=np.array([[0,1],[-1,0]])
        return np.kron(np.eye(self.L),Omega)

    def measure(self,n_list,ix):
        ''' Majorana site index for ix'''
        if not hasattr(self,'n_history'):
            self.n_history=[]
        if not hasattr(self,'i_history'):
            self.i_history=[]

        m=self.C_m_history[-1].copy()
        
        # swap(m,ix)
        proj=[self.kraus(n) for n in n_list]
        ix_bar=np.array([i for i in np.arange(self.L*2) if i not in ix])
        Psi=_contraction(m,proj,ix,ix_bar)
        assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))

        # for i_ind,i in enumerate(ix):
        #     Psi[[i,-(len(ix)-i_ind)]]=Psi[[-(len(ix)-i_ind),i]]
        #     Psi[:,[i,-(len(ix)-i_ind)]]=Psi[:,[-(len(ix)-i_ind),i]]
        # swap(Psi,ix)
        
        if self.history:
            self.C_m_history.append(Psi)
            self.n_history.append(n_list)
            self.i_history.append(ix)
        else:
            self.C_m_history=[Psi]
            self.n_history=[n_list]
            self.i_history=[ix]


    def projection(self,s):
        '''
        occupancy number: s= 0,1 
        (-1)^0 even parity, (-1)^1 odd parity

        '''
        assert (s==0 or s==1),"s={} is either 0 or 1".format(s)
        blkmat=np.array([[0,-(-1)**s,0,0],
                        [(-1)**s,0,0,0],
                        [0,0,0,(-1)**s],
                        [0,0,-(-1)**s,0]])
        return blkmat

    def kraus(self,n):
        return -np.array([[0,n[0],n[1],n[2]],
                        [-n[0],0,-n[2],n[1]],
                        [-n[1],n[2],0,-n[0]],
                        [-n[2],-n[1],n[0],0]])
    
    def mutual_information_m(self,subregion_A,subregion_B):
        ''' Composite fermion site index'''
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        s_A=self.von_Neumann_entropy_m(subregion_A)
        s_B=self.von_Neumann_entropy_m(subregion_B)
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy_m(subregion_AB)
        return s_A+s_B-s_AB

    def von_Neumann_entropy_m(self,subregion):
        c_A=self.c_subregion_m(subregion)
        val=nla.eigvalsh(1j*c_A)
        self.val_sh=val
        val=np.sort(val)
        val=(1-val)/2+1e-18j   #\lambda=(1-\xi)/2
        return np.real(-np.sum(val*np.log(val))-np.sum((1-val)*np.log(1-val)))/2

    def c_subregion_m(self,subregion,Gamma=None):
        if not hasattr(self,'C_m'):
            self.covariance_matrix_f()
        if Gamma is None:
            Gamma=self.C_m_history[-1]
        subregion=linearize_index(subregion,2)
        return Gamma[np.ix_(subregion,subregion)]


    def measure_all(self):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1


    def measure_all(self,a1,a2,b1,b2,even=True,n1_z=True,Born=False):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        if Born:
            Gamma_list=self.C_m_history[-1][proj_range,(proj_range+1)%(2*self.L)]
            n_list=get_Born(a1,a2,b1,b2,Gamma_list,n1_z=n1_z,rng=self.rng)
        else:
            n_list=get_random(a1,a2,b1,b2,proj_range.shape[0],n1_z=n1_z,rng=self.rng)
        for i,n in zip(proj_range,n_list):
            self.measure([n], np.array([i,(i+1)%(2*self.L)]))

    def measure_all_sync(self,a1,a2,b1,b2,even=True,n1_z=True,Born=False):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        if Born:
            Gamma_list=self.C_m_history[-1][proj_range,(proj_range+1)%(2*self.L)]
            n_list=get_Born(a1,a2,b1,b2,Gamma_list,n1_z=n1_z,rng=self.rng)
        else:
            n_list=get_random(a1,a2,b1,b2,proj_range.shape[0],n1_z=n1_z,rng=self.rng)
        
        self.measure(n_list,np.c_[proj_range,(proj_range+1)%(2*self.L)].flatten())

    def mutual_information_cross_ratio(self):
        x=np.array([0,self.L//4,self.L//2,self.L//4*3])
        MI=[]
        subA=np.arange(x[0],x[1])
        subB=np.arange(x[2],x[3])
        for shift in range(self.L//2):
            MI.append(self.mutual_information_m((subA+shift)%self.L, (subB+shift)%self.L))
        return np.mean(MI)

def get_random(a1,a2,b1,b2,num,n1_z=True,rng=None):
    '''
        -b1<-a1<a2<b2 
        
        n1=True: nA=(n1,n2,n3)
        n1=True: nB=(n3,n1,n2)
    '''
    assert -b1<=-a1<=a2<=b2, "the order of -b1<-a1<a2<b2 not satisfied"
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    # inverse of CDF
    n1=np.where(sign<.5,sign*2*(b1-a1)-b1,(sign-1/2)*2*(b2-a2)+a2)

    # use rescale
    # n1=np.where(sign<0.5,rescale(sign,y0=-b1,y1=-a1,x0=0,x1=.5),rescale(sign,y0=a2,y1=b2,x0=.5,x1=1))
    # complete random
    # n1=np.random.uniform(b2,b1-a1+a2,num)
    # n1=np.where(n1<a2,n1,n1+(a1-a2))

    n2,n3=get_inplane(n1, num,rng=rng)
    return np.c_[n1,n2,n3] if n1_z else np.c_[n3,n1,n2]


def get_inplane(n1,num,rng=None):
    r=np.sqrt(1-n1**2)
    rng=np.random.default_rng(rng)
    phi=rng.random(num)*2*np.pi
    n2,n3=r*np.cos(phi),r*np.sin(phi)
    return n2,n3

def get_Born(a1,a2,b1,b2,Gamma,n1_z=True,rng=None):
    '''
        -b1<-a1<a2<b2 
        Gamma: list for all parities
        
        n1=True: nA=(n1,n2,n3)
        n1=True: nB=(n3,n1,n2)
    '''
    assert -b1<=-a1<=a2<=b2, "the order of -b1<-a1<a2<b2 not satisfied"
    num=Gamma.shape[0]
    rng=np.random.default_rng(rng)
    u=rng.random(size=num)
    theta1,theta2=(a2+b2)/((b1-a1)*(a1+a2+b1+b2)),(a1+b1)/((b2-a2)*(a1+a2+b1+b2))
    bndy=theta1*(-a1+b1+1/2*(a1**2-b1**2)*Gamma)
    coef1=1/2*theta1*Gamma,theta1,theta1*b1-1/2*theta1*b1**2*Gamma
    coef2=1/2*theta2*Gamma,theta2,theta1*(-a1+b1+(a1**2-b1**2)*Gamma/2)-a2**2*Gamma*theta2/2-theta2*a2

    n1=np.where(u<bndy,solve(coef1,u),solve(coef2,u))

    n2,n3=get_inplane(n1, num,rng=rng)
    return np.c_[n1,n2,n3] if n1_z else np.c_[n3,n1,n2]

def solve(coef,u):
    a,b,c=coef
    c=c-u
    with np.errstate(invalid='ignore'):
        n1=np.where(a==0,-c/b,(-b+np.sqrt(b**2-4*a*c))/(2*a) )
    return n1
    # return -c/b if a==0 else (-b+np.sqrt(b**2-4*a*c))/(2*a) 


def rescale(x,y0,y1,x0=0,x1=1):
    return (y1-y0)/(x1-x0)*(x-x0)+y0

def cross_ratio(x,L):
    if L<np.inf:
        xx=lambda i,j: (np.sin(np.pi/(L)*np.abs(x[i]-x[j])))
    else:
        xx=lambda i,j: np.abs(x[i]-x[j])
    eta=(xx(0,1)*xx(2,3))/(xx(0,2)*xx(1,3))
    return eta

# @jit(float64[:,:](float64[:,:],float64[:,:],int64[:]),nopython=True,fastmath=True)
def _contraction(m,proj_list,ix,ix_bar):
    ix,ix_bar=list(ix),list(ix_bar)
    # Gamma_LL=m[:-len(ix),:-len(ix)]
    # Gamma_LR=m[:-len(ix),-len(ix):]
    # Gamma_RR=m[-len(ix):,-len(ix):]
    Gamma_LL=m[np.ix_(ix_bar,ix_bar)]
    Gamma_LR=m[np.ix_(ix_bar,ix)]
    Gamma_RR=m[np.ix_(ix,ix)]

    proj=np.zeros((4*len(proj_list),4*len(proj_list)))
    # change index from (in_1, in_2, out_1, out_2) (in_3, in_4, out_3, out_4)
    # to (in_1 , in_2, in_3, in_4, out_1, out_2, out_3, out_4)
    for i,p in enumerate(proj_list):
        proj[np.ix_([2*i,2*i+1,2*i+2*len(proj_list),2*i+2*len(proj_list)+1],[2*i,2*i+1,2*i+2*len(proj_list),2*i+2*len(proj_list)+1])]=p
    
    Upsilon_LL=proj[:len(ix),:len(ix)]
    Upsilon_RR=proj[len(ix):,len(ix):]
    Upsilon_RL=proj[len(ix):,:len(ix)]

    # zero=np.zeros((m.shape[0]-len(ix),len(ix)))
    # zero0=np.zeros((len(ix),len(ix)))f
    # mat1=np.block([[Gamma_LL,zero],[zero.T,Upsilon_RR]])
    # mat2=np.block([[Gamma_LR,zero],[zero0,Upsilon_RL]])
    # mat3=np.block([[Gamma_RR,np.eye(len(ix))],[-np.eye(len(ix)),Upsilon_LL]])

    mat1,mat2,mat3=np.zeros(m.shape),np.zeros((m.shape[0],2*len(ix))),np.zeros((2*len(ix),2*len(ix)))
    mat1[:-len(ix),:-len(ix)]=Gamma_LL
    mat1[-len(ix):,-len(ix):]=Upsilon_RR
    mat2[:-len(ix),:len(ix)]=Gamma_LR
    mat2[-len(ix):,-len(ix):]=Upsilon_RL
    mat3[:len(ix),:len(ix)]=Gamma_RR
    mat3[len(ix):,len(ix):]=Upsilon_LL
    mat3[:len(ix),len(ix):]=np.eye(len(ix))
    mat3[len(ix):,:len(ix)]=-np.eye(len(ix))

    if np.count_nonzero(mat2):
        Psi=mat1+mat2@(nla.solve(mat3,mat2.T))
        # Psi=mat1+mat2@nla.inv(mat3)@mat2.T
        # Psi=mat1+mat2@(la.lstsq(mat3,mat2.T)[0])
    else:
        print('mat2 is singular')
        Psi=mat1
    
    Psi_mat=np.zeros_like(Psi)
    Psi_mat[np.ix_(ix_bar,ix_bar)]=Psi[:len(ix_bar),:len(ix_bar)]
    Psi_mat[np.ix_(ix,ix)]=Psi[-len(ix):,-len(ix):]
    Psi_mat[np.ix_(ix_bar,ix)]=Psi[:len(ix_bar),-len(ix):]
    Psi_mat[np.ix_(ix,ix_bar)]=Psi[-len(ix):,:len(ix_bar)]
    Psi=Psi_mat

    Psi=(Psi-Psi.T)/2
    return Psi

def swap(mat,ix):
    '''Swap ix with last ix on both col and row'''
    for i_ind,i in enumerate(ix):
        mat[[i,-(len(ix)-i_ind)]]=mat[[-(len(ix)-i_ind),i]]
        mat[:,[i,-(len(ix)-i_ind)]]=mat[:,[-(len(ix)-i_ind),i]]

def linearize_index(subregion,n,k=2,proj=False):
    try:
        subregion=np.array(subregion)
    except:
        raise ValueError("The subregion is ill-defined"+subregion)
    if proj:
        return sorted(np.concatenate([n*subregion+i for i in range(0,n,k)]))
    else:
        return sorted(np.concatenate([n*subregion+i for i in range(n)]))
