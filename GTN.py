import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
from copy import copy

class GTN:
    def __init__(self,L,history=True,seed=None,op=False,random_init=False,c=[1,1,1]):
        self.L=L
        self.op=op
        self.random_init=random_init
        self.rng=np.random.default_rng(seed)
        self.C_m=self.correlation_matrix(op=op)
        self.C_m_history=[self.C_m]
        self.history=history
        self.n_history=[]
        self.i_history=[]
        self.MI_history=[]
        self.c=c

    
    def correlation_matrix(self,op):
        if op:
            return np.zeros((0,0))
        else:
            Omega=np.array([[0,1.],[-1.,0]])
            Omega_diag=np.kron(np.eye(self.L),Omega)
            O=get_O(self.rng,2*self.L) if self.random_init else np.eye(2*self.L)
            Gamma=O@Omega_diag@O.T
            return (Gamma-Gamma.T)/2
            

    def measure(self,n_list,ix):
        ''' Majorana site index for ix'''

        m=self.C_m_history[-1].copy()
        proj=[self.kraus(n) for n in n_list]
        ix_bar=np.array([i for i in np.arange(self.L*2) if i not in ix]) if not self.op else np.array([i for i in np.arange(self.L*4) if i not in ix])
        Psi=_contraction(m,proj,ix,ix_bar)
        assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))
        if self.history:
            self.C_m_history.append(Psi)
            self.n_history.append(n_list)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            self.C_m_history=[Psi]
            self.n_history=[n_list]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]


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
        c=self.c
        return np.array([[0,c[0]*n[0],c[1]*n[1],c[2]*n[2]],
                        [-c[0]*n[0],0,-c[2]*n[2],c[1]*n[1]],
                        [-c[1]*n[1],c[2]*n[2],0,-c[0]*n[0]],
                        [-c[2]*n[2],-c[1]*n[1],c[0]*n[0],0]])
        # return -np.array([[0,n[0],-n[1],n[2]],
        #                 [-n[0],0,-n[2],-n[1]],
        #                 [n[1],n[2],0,-n[0]],
        #                 [-n[2],n[1],n[0],0]])
        # return -np.array([[0,n[0],n[1],n[2]],
        #                 [-n[0],0,-n[2],n[1]],
        #                 [-n[1],n[2],0,-n[0]],
        #                 [-n[2],-n[1],n[0],0]])
    

    def measure_all(self,a1,a2,b1,b2,even=True,theta_list=0,phi_list=0,Born=False):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        if Born:
            Gamma_list=self.C_m_history[-1][proj_range,(proj_range+1)%(2*self.L)]
            n_list=get_Born(a1,a2,b1,b2,Gamma_list,theta_list=theta_list,phi_list=phi_list,rng=self.rng)
        else:
            n_list=get_random(a1,a2,b1,b2,proj_range.shape[0],theta_list=theta_list,phi_list=phi_list,rng=self.rng)
        for i,n in zip(proj_range,n_list):
            self.measure([n], np.array([i,(i+1)%(2*self.L)]))

    def measure_all_sync(self,a1,a2,b1,b2,even=True,theta_list=0,phi_list=0,Born=False):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        proj_range_1=proj_range if not self.op else proj_range+2* self.L
        proj_range_2=(proj_range+1)%(2*self.L) if not self.op else (proj_range+1)%(2*self.L) + 2*self.L
        if isinstance(theta_list, int) or isinstance(theta_list, float):
            theta_list=[theta_list]*len(proj_range_1)
        if isinstance(phi_list, int) or isinstance(phi_list, float):
            phi_list=[phi_list]*len(proj_range_1)
        if Born:
            if self.C_m_history[-1].size==0:
                Gamma_list=np.array([1]*self.L)
                n_list=get_Born(a1,a2,b1,b2,Gamma_list,theta_list=theta_list,phi_list=phi_list,rng=self.rng)
                self.measure(n_list,np.c_[proj_range_1,proj_range_2].flatten())
            else:
                for i,j in zip(proj_range_1,proj_range_2):
                    Gamma=self.C_m_history[-1][[i],[j]]
                    if even:
                        n_list=get_Born_A(a1,a2,b1,b2,Gamma,rng=self.rng)
                    else:
                        n_list=get_Born_B(a1,a2,b1,b2,Gamma,rng=self.rng)
                    self.measure(n_list,[i,j])
        else:
            n_list=get_random(a1,a2,b1,b2,proj_range.shape[0],theta_list=theta_list,phi_list=phi_list,rng=self.rng)
            self.measure(n_list,np.c_[proj_range_1,proj_range_2].flatten())

    def measure_all_Haar(self,sigma=0,even=True,theta_list=0,phi_list=0):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        C_m=self.C_m_history[-1]
        proj_range_1=proj_range
        proj_range_2=(proj_range+1)%(2*self.L)
        if isinstance(theta_list, int) or isinstance(theta_list, float):
            theta_list=[theta_list]*len(proj_range_1)
        if isinstance(phi_list, int) or isinstance(phi_list, float):
            phi_list=[phi_list]*len(proj_range_1)
        n_list=get_Haar(sigma,proj_range.shape[0],rng=self.rng,theta_list=theta_list,phi_list=phi_list)
        self.measure(n_list,np.c_[proj_range_1,proj_range_2].flatten())

    def measure_all_tri_op(self,p,Born=False,even=True):
        '''The Kraus operator is composed of only three
        sqrt(1-p) exp(i*phi* i* gamma_i * gamma_i+1), phi ~ U[0,2pi]; n=(0,cos(phi),sin(phi))
        sqrt(p) (1+i* gamma_i * gamma_i+1)/2; n=(-1,0,0)
        sqrt(p) (1-i* gamma_i * gamma_i+1)/2; n=(1,0,0)

        For n_A, we literally take p, while for n_B we substitute the prob from p to 1-p
        '''
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        proj_range_1=proj_range if not self.op else proj_range+2* self.L
        proj_range_2=(proj_range+1)%(2*self.L) if not self.op else (proj_range+1)%(2*self.L) + 2*self.L
        if Born:
            if self.C_m_history[-1].size==0:
                pass
            else:
                for i,j in zip(proj_range_1,proj_range_2):
                    Gamma=self.C_m_history[-1][[i],[j]]
                    n_list=get_Born_tri_op(p,Gamma,rng=self.rng)
                    self.measure(n_list,[i,j])
        else:
            n_list=get_random_tri_op(p,proj_range.shape[0],rng=self.rng)
            self.measure(n_list,np.c_[proj_range_1,proj_range_2].flatten())


    def mutual_information_cross_ratio(self,ratio=[1,4]):
        
        x=np.array([0,self.L//ratio[1]*ratio[0],self.L//2,self.L//2+self.L//ratio[1]*ratio[0]])
        # x=np.array([0,self.L//8,self.L//2,self.L//8*5])
        MI=[]
        subA=np.arange(x[0],x[1])
        subB=np.arange(x[2],x[3])
        for shift in np.arange(0,self.L//2):
            MI.append(self.mutual_information_m((subA+shift)%self.L, (subB+shift)%self.L))
        return np.mean(MI)
        # return MI

    def entanglement_contour(self,subregion):
        c_A=self.c_subregion_m(subregion)
        C_f=(np.eye(c_A.shape[0])+1j*c_A)/2
        f,_=la.funm(C_f,lambda x: -x*np.log(x),disp=False)
        return np.diag(f).real.reshape((-1,2)).sum(axis=1).real

    def mutual_information_m(self,subregion_A,subregion_B,Gamma=None):
        ''' Composite fermion site index'''
        assert np.intersect1d(subregion_A,subregion_B).size==0 , "Subregion A and B overlap"
        s_A=self.von_Neumann_entropy_m(subregion_A,Gamma)
        s_B=self.von_Neumann_entropy_m(subregion_B,Gamma)
        subregion_AB=np.concatenate([subregion_A,subregion_B])
        s_AB=self.von_Neumann_entropy_m(subregion_AB,Gamma)
        return s_A+s_B-s_AB

    def von_Neumann_entropy_m(self,subregion,Gamma=None):
        c_A=self.c_subregion_m(subregion,Gamma)
        val=nla.eigvalsh(1j*c_A)
        # self.val_sh=val
        val=np.sort(val)
        val=(1-val)/2+1e-18j   #\lambda=(1-\xi)/2
        return np.real(-np.sum(val*np.log(val))-np.sum((1-val)*np.log(1-val)))/2

    def c_subregion_m(self,subregion,Gamma=None):
        if Gamma is None:
            Gamma=self.C_m_history[-1]
        subregion=self.linearize_index(subregion,2)
        return Gamma[np.ix_(subregion,subregion)]

    def linearize_index(self,subregion,n,k=2,proj=False):
        try:
            subregion=np.array(subregion)
        except:
            raise ValueError("The subregion is ill-defined"+subregion)
        if proj:
            return np.int_(sorted(np.concatenate([n*subregion+i for i in range(0,n,k)])))%(2*self.L)
        else:
            return np.int_(sorted(np.concatenate([n*subregion+i for i in range(n)])))%(2*self.L)
        
def get_random_tri_op(p,num,rng=None):
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    n1= (sign<p/2)*(-1)+(sign>1-p/2)
    n2,n3=get_inplane(n1, num,rng=rng,)
    return np.c_[n1,n2,n3]

def get_Born_tri_op(p,Gamma,rng=None):
    num=Gamma.shape[0]
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    n1= (sign<p*(1+Gamma)/2)*(-1)+(sign>p*(1+Gamma)/2+1-p)
    n2,n3=get_inplane(n1, num,rng=rng)
    return np.c_[n1,n2,n3]

def get_random(a1,a2,b1,b2,num,rng=None,theta_list=0,phi_list=0):
    '''
        -b1<-a1<a2<b2 
        
        n1=True: nA=(n1,n2,n3)
        n1=True: nB=(n3,n1,n2)
    '''
    assert -b1<=-a1<=a2<=b2, "the order of -b1<-a1<a2<b2 not satisfied"
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    k=1/(b1-a1+b2-a2)
    n1=np.where(sign<(b1-a1)*k,sign/k-b1,(sign-k*(b1-a1))/k+a2)

    # inverse of CDF
    # n1=np.where(sign<.5,sign*2*(b1-a1)-b1,(sign-1/2)*2*(b2-a2)+a2)

    # use rescale
    # n1=np.where(sign<0.5,rescale(sign,y0=-b1,y1=-a1,x0=0,x1=.5),rescale(sign,y0=a2,y1=b2,x0=.5,x1=1))
    # complete random
    # n1=np.random.uniform(b2,b1-a1+a2,num)
    # n1=np.where(n1<a2,n1,n1+(a1-a2))

    n2,n3=get_inplane(n1, num,rng=rng)
    n=np.c_[n1,n2,n3]
    return rotate(n,theta_list,phi_list)

def get_O(rng,n):
    rng=np.random.default_rng(rng)
    A=rng.normal(size=(n,n))
    AA=(A-A.T)/2
    return la.expm(AA)

def rotate(n,theta,phi):
    n=np.c_[np.cos(theta)*n[:,0]-np.sin(theta)*n[:,1],np.sin(theta)*n[:,0]+np.cos(theta)*n[:,1],n[:,2]]
    n=np.c_[np.cos(phi)*n[:,0]+np.sin(phi)*n[:,2],n[:,1],-np.sin(phi)*n[:,0]+np.cos(phi)*n[:,2]]
    return n


def get_inplane(n1,num,rng=None,sigma=1):
    r=np.sqrt(1-n1**2)
    rng=np.random.default_rng(rng)
    phi=rng.random(num)*2*np.pi*sigma
    n2,n3=r*np.cos(phi),r*np.sin(phi)
    return n2,n3

def get_Born_A(a1,a2,b1,b2,Gamma,rng=None,):
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
    bndy=theta1*(-a1+b1-1/2*(a1**2-b1**2)*Gamma)
    coef1=-1/2*theta1*Gamma,theta1,theta1*b1+1/2*theta1*b1**2*Gamma
    coef2=-1/2*theta2*Gamma,theta2,theta1*(-a1+b1-(a1**2-b1**2)*Gamma/2)+a2**2*Gamma*theta2/2-theta2*a2

    n1=np.where(u<bndy,solve(coef1,u),solve(coef2,u))

    n2,n3=get_inplane(n1, num,rng=rng)
    n=np.c_[n1,n2,n3]
    return n

    # return rotate(n,theta_list,phi_list)

def get_Born_B(a1,a2,b1,b2,Gamma,rng=None):
    '''
        -b1<-a1<a2<b2 
        Gamma: list for all parities
        
        n1=True: nA=(n1,n2,n3)
        n1=True: nB=(n3,n1,n2)
    '''
    assert -b1<=-a1<=a2<=b2, "the order of -b1<-a1<a2<b2 not satisfied"
    # num=Gamma.shape[0]
    rng=np.random.default_rng(rng)

    s=get_random(a1,a2,b1,b2,num=1,rng=rng)[0,0]

    # w1=(a2+b2)/((b1-a1)*(a1+a2+b1+b2))
    # w2=(a1+b1)/((b2-a2)*(a1+a2+b1+b2))
    # bndy=w1*(b1-a1)
    # u=rng.random()
    # s=np.where(u<bndy,u/w1-b1,(u-w1*(b1-a1))/w2+a2)

    phi=get_random_phi(s,Gamma[0],rng.random())
    # return s,phi
    return np.array([[np.sin(phi)*np.sqrt(1-s**2),s,np.cos(phi)*np.sqrt(1-s**2)]])

def get_random_phi(s,Gamma,u):
    phi=0
    while np.abs((phi-np.sqrt(1-s**2)*Gamma*(1-np.cos(phi)))/(2*np.pi)-u)>1e-8:
        phi=u*2*np.pi+np.sqrt(1-s**2)*Gamma*(1-np.cos(phi))
    return phi

def get_Haar(sigma,num,rng=None,theta_list=0,phi_list=0):
    rng=np.random.default_rng(rng)
    u=rng.random(size=num)
    n1=rescale(u, 1-2*sigma, 1)
    n2,n3=get_inplane(n1, num,rng=rng)
    n=np.c_[n1,n2,n3]
    return rotate(n,theta_list,phi_list)

def solve(coef,u):
    a,b,c=coef
    c=c-u
    with np.errstate(invalid='ignore'):
        n1=np.where(a==0,-c/b,(-b+np.sqrt(b**2-4*a*c))/(2*a) )
    return n1


def rescale(x,y0,y1,x0=0,x1=1):
    return (y1-y0)/(x1-x0)*(x-x0)+y0

def cross_ratio(x,L):
    if L<np.inf:
        xx=lambda i,j: (np.sin(np.pi/(L)*np.abs(x[i]-x[j])))
    else:
        xx=lambda i,j: np.abs(x[i]-x[j])
    eta=(xx(0,1)*xx(2,3))/(xx(0,2)*xx(1,3))
    return eta

def cord(x,L):
    return L/np.pi*np.sin(np.pi/L*np.abs(x))

# @jit(float64[:,:](float64[:,:],float64[:,:],int64[:]),nopython=True,fastmath=True)
def _contraction(m,proj_list,ix,ix_bar):
    ix,ix_bar=list(ix),list(ix_bar)

    proj=np.zeros((4*len(proj_list),4*len(proj_list)))
    # change index from (in_1, in_2, out_1, out_2) (in_3, in_4, out_3, out_4)
    # to (in_1 , in_2, in_3, in_4, out_1, out_2, out_3, out_4)
    for i,p in enumerate(proj_list):
        proj[np.ix_([2*i,2*i+1,2*i+2*len(proj_list),2*i+2*len(proj_list)+1],[2*i,2*i+1,2*i+2*len(proj_list),2*i+2*len(proj_list)+1])]=p

    if m.size==0:
        return proj
    else:
        Gamma_LL=m[np.ix_(ix_bar,ix_bar)]
        Gamma_LR=m[np.ix_(ix_bar,ix)]
        Gamma_RR=m[np.ix_(ix,ix)]

    Upsilon_LL=proj[:len(ix),:len(ix)]
    Upsilon_RR=proj[len(ix):,len(ix):]
    Upsilon_RL=proj[len(ix):,:len(ix)]

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
    return (Psi)


def interpolate(x1,x2,l0,h0,L,k=1):
    x=np.arange(L)
    h=h0/2
    l=l0-h0/2
    return (h-l)/2*(np.tanh((x-x1)*k)+1)+l-(h-l)/2*(np.tanh((x-x2)*k)+1)+h
