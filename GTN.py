# %%writefile GTN.py 
import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
from scipy.sparse import bsr_array
from copy import copy
from itertools import permutations

class GTN:
    def __init__(self,L,history=True,seed=None,op=False,random_init=False,c=[1,1,1],pbc=True,trijunction=False):
        self.L=L
        self.op=op
        self.trijunction=trijunction
        self.random_init=random_init
        self.rng=np.random.default_rng(seed)
        self.C_m=self.correlation_matrix()
        self.Gamma_like=np.zeros_like(self.C_m)
        self.C_m_history=[self.C_m]
        self.history=history
        self.n_history=[]
        self.i_history=[]
        self.MI_history=[]
        self.p_history=[]
        self.c=c
        self.pbc=pbc
        self.full_ix=set(range(self.C_m[-1].shape[0]))

    
    def correlation_matrix(self):
        '''if `self.op` is on, the computational chain will duplicate a reference chain, where both sites (i,i+2L) are coupled, with parity +1'''
        if self.trijunction:
            '''construct trijuction, the order of the three chain are arranged from outmost to innermost, namely, 0->2L-1 is Chain 0; 2L->4L-1 is Chain 1; 4L->6L-1 is Chain 2'''
            if self.op:
                Gamma=np.zeros((12*self.L,12*self.L))
                EPR=np.fliplr(np.diag([1,-1]))
                for i in range(6*self.L):
                    Gamma[np.ix_([i,i+6*self.L],[i,i+6*self.L])]=EPR
            else:
                Omega=np.array([[0,1.],[-1.,0]])
                Omega_diag=np.kron(np.eye(self.L),Omega)
                Gamma=np.kron(np.eye(3),Omega_diag)
        else:
            '''construct normal 1d chain'''
            if self.op:
                Gamma=np.zeros((4*self.L,4*self.L))
                EPR=np.fliplr(np.diag([1,-1]))
                for i in range(2*self.L):
                    Gamma[np.ix_([i,i+2*self.L],[i,i+2*self.L])]=EPR
            else:
                Omega=np.array([[0,1.],[-1.,0]])
                Omega_diag=np.kron(np.eye(self.L),Omega)
                self.A_D=Omega_diag
                O=get_O(self.rng,2*self.L) if self.random_init else np.eye(2*self.L)
                Gamma=O@Omega_diag@O.T
        return (Gamma-Gamma.T)/2
            
    def set(self,ij_list,n_list):
        """ij_list: [[i,j],...]
        n_list: [1,-1,...]
        simply set all Gamma[i,j]=n, Gamma[j,i]=-n
        """
        Gamma=self.C_m_history[-1]
        for ij,n in zip(ij_list,n_list):
            i,j=ij
            Gamma[i,j]=n
            Gamma[j,i]=-n

    def measure(self,n,ix):
        ''' Majorana site index for ix'''

        Psi=self.C_m_history[-1]
        proj=self.kraus(n)
        # ix_bar=np.array([i for i in np.arange(self.L*2) if i not in ix]) if not self.op else np.array([i for i in np.arange(self.L*4) if i not in ix])
        ix_bar=np.array([i for i in np.arange(self.C_m[-1].shape[0]) if i not in ix])
        # Psi=P_contraction(m,proj,ix,ix_bar)
        P_contraction_2(Psi,proj,ix,ix_bar,self.Gamma_like,reset_Gamma_like=False)

        # assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))
        if self.history:
            self.C_m_history.append(Psi)
            self.n_history.append(n)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            self.C_m_history=[Psi]
            self.n_history=[n]
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
    
    def op_class_AIII(self,A,theta1,theta2,kind):
        Gamma=np.zeros((8,8),dtype=float)
        assert 0<=A<=1, "A should be within [0,1]"
        sq_A=np.sqrt(1-A**2)
        cos_theta1,sin_theta1=np.cos(theta1),np.sin(theta1)
        cos_theta2,sin_theta2=np.cos(theta2),np.sin(theta2)
        if kind == (1,1) or kind == (-1,-1):
            Gamma[0,1]=Gamma[2,3]=kind[0]*A
            Gamma[4,5]=Gamma[6,7]=-kind[0]*A
            Gamma[0,4]=Gamma[1,5]=sq_A*cos_theta1
            Gamma[0,5]=-sq_A*sin_theta1
            Gamma[1,4]=sq_A*sin_theta1
            Gamma[2,6]=Gamma[3,7]=sq_A*cos_theta2
            Gamma[2,7]=-sq_A*sin_theta2
            Gamma[3,6]=sq_A*sin_theta2
        elif kind == (-1,1) or kind == (1,-1):
            cos_theta1_theta2,sin_theta1_theta2=np.cos(theta1-theta2),np.sin(theta1-theta2)
            Gamma[0,3]=kind[0]*A*cos_theta1_theta2
            Gamma[5,6]=kind[0]*A
            Gamma[1,2]=-kind[0]*A*cos_theta1_theta2
            Gamma[4,7]=-kind[0]*A
            Gamma[0,4]=Gamma[1,5]=sq_A*cos_theta1
            Gamma[0,5]=-sq_A*sin_theta1
            Gamma[1,4]=sq_A*sin_theta1
            Gamma[0,2]=Gamma[1,3]=kind[0]*A*sin_theta1_theta2
            Gamma[2,6]=Gamma[3,7]=sq_A*cos_theta2
            Gamma[2,7]=-sq_A*sin_theta2
            Gamma[3,6]=sq_A*sin_theta2
        else:
            raise ValueError(f'kind {kind} not defined')
        return (Gamma-Gamma.T)

    def measure_class_AIII(self,A,theta1,theta2,kind,ix,):
        ''' Majorana site index for ix'''
        assert len(ix)==4, 'len of ix should be 4'
        Psi=self.C_m_history[-1]
        ix_bar=np.array(list(self.full_ix-set(ix)))
        proj=self.op_class_AIII(A,theta1,theta2,kind)
        P_contraction_2(Psi,proj,ix,ix_bar,self.Gamma_like,reset_Gamma_like=False)
        if self.history:
            self.C_m_history.append(Psi.copy())
            self.n_history.append([A,theta1,theta2,kind])
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            # self.C_m_history=[Psi]
            self.n_history=[A,theta1,theta2,kind]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]

    def measure_all(self,a1,a2,b1,b2,even=True,theta_list=0,phi_list=0,Born=False):
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1 # Majorana site index of left leg
        if Born:
            Gamma_list=self.C_m_history[-1][proj_range,(proj_range+1)%(2*self.L)]
            n_list=get_Born(a1,a2,b1,b2,Gamma_list,theta_list=theta_list,phi_list=phi_list,rng=self.rng)
        else:
            n_list=get_random(a1,a2,b1,b2,proj_range.shape[0],theta_list=theta_list,phi_list=phi_list,rng=self.rng)
        for i,n in zip(proj_range,n_list):
            self.measure([n], np.array([i,(i+1)%(2*self.L)]))
    
    def measure_all_class_AIII(self,A_list,Born=True,class_A=False,even=True,):
        proj_range=np.arange(self.L//2)*4 if even else np.arange(self.L//2)*4+2 # Majorana site index of leftmost leg
        if isinstance(A_list, int) or isinstance(A_list, float):
            A_list=np.array([A_list]*len(proj_range))
        if self.history:
            self.p_history.append(A_list)
        else:
            self.p_history=[A_list]
        if Born:
            for i, A in zip(proj_range,A_list):
                legs=[i,(i+1)%(2*self.L),(i+2)%(2*self.L),(i+3)%(2*self.L)]
                Gamma=self.C_m_history[-1][np.ix_(legs,legs)]
                kind,theta1,theta2=get_Born_class_AIII(A=A,Gamma=Gamma,rng=self.rng,class_A=class_A,)
                self.measure_class_AIII(A=A,theta1=theta1,theta2=theta2,kind=kind,ix=legs)
        else:
            pass
    
    def measure_all_class_AIII_r(self,A_list,r_list,Born=True,class_A=False,intraleg=True,):
        site_A_left=np.arange(self.L//2)*4
        site_B_left=np.arange(self.L//2)*4+2
        if isinstance(A_list, int) or isinstance(A_list, float):
            A_list=np.array([A_list]*(self.L//2))
        if isinstance(r_list, int) or isinstance(r_list, float):
            r_list=np.array([r_list]*(self.L//2))
        if self.history:
            self.p_history.append(A_list)
        else:
            self.p_history=[A_list]
        if Born:
            for idx in range(self.L//2):
                r0=int(np.round(self.rng.uniform(r_list[idx]-1/2,r_list[idx]+1/2)))
                if intraleg:
                    legs=[site_B_left[idx],(site_B_left[idx]+1)%(2*self.L),site_B_left[(idx+r0)%(self.L//2)],(site_B_left[(idx+r0)%(self.L//2)]+1)%(2*self.L)]
                else:
                    legs=[site_B_left[idx],(site_B_left[idx]+1)%(2*self.L),site_A_left[(idx+r0)%(self.L//2)],(site_A_left[(idx+r0)%(self.L//2)]+1)%(2*self.L)]

                if r0 ==0 and intraleg:
                    # tackle onsite unitary
                    pass
                else:
                    Gamma=self.C_m_history[-1][np.ix_(legs,legs)]
                    kind,theta1,theta2=get_Born_class_AIII(A=A_list[idx],Gamma=Gamma,rng=self.rng,class_A=class_A,)
                    self.measure_class_AIII(A=A_list[idx],theta1=theta1,theta2=theta2,kind=kind,ix=legs)
        else:
            pass
        

        

    def measure_all_sync(self,a1,a2,b1,b2,even=True,theta_list=0,phi_list=0,Born=False):
        """sync means all operators apply the layer at the same time"""
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

    def measure_all_tri_op(self,p_list,Born=False,even=True):
        '''The Kraus operator is composed of only three
        sqrt(1-p) exp(i*phi* i* gamma_i * gamma_i+1), phi ~ U[0,2pi]; n=(0,cos(phi),sin(phi))
        sqrt(p) (1+i* gamma_i * gamma_i+1)/2; n=(-1,0,0)
        sqrt(p) (1-i* gamma_i * gamma_i+1)/2; n=(1,0,0)

        For n_A, we literally take p, while for n_B we substitute the prob from p to 1-p
        '''
        proj_range=np.arange(self.L)*2 if even else np.arange(self.L)*2+1
        # Why do you want to change the system sites to the later part?
        # proj_range_1=proj_range if not self.op else proj_range+2* self.L
        # proj_range_2=(proj_range+1)%(2*self.L) if not self.op else (proj_range+1)%(2*self.L) + 2*self.L
        proj_range_1=proj_range
        proj_range_2=(proj_range+1)%(2*self.L)
        if isinstance(p_list, int) or isinstance(p_list, float):
            p_list=np.array([p_list]*len(proj_range_1))
        if self.history:
            self.p_history.append(p_list)
        else:
            self.p_history=[p_list]
        if Born:
            for i,j,p in zip(proj_range_1,proj_range_2,p_list):
                Gamma=self.C_m_history[-1][[i],[j]]
                n_list=get_Born_tri_op(p,Gamma,rng=self.rng)
                if not self.pbc and not even and i==proj_range_1[-1]:
                    continue
                self.measure(n_list[0],[i,j])
        else:
            n_list=get_random_tri_op(p_list,proj_range.shape[0],rng=self.rng)
            for i,j,n in zip(proj_range_1,proj_range_2,n_list):
                try:
                    self.measure(n,[i,j])
                except:
                    n[0]=-n[0]
                    # This is a workaround to let it run, however, in forced measurement, the strong projection is problematic, as it can sometimes vanish the state
                    self.measure(n,[i,j])

    def measure_list_tri_op(self,site_list,p_list,Born=True,):
        '''site_list: [[i1,j1],[i2,j2],[i3,j3],...] measures [i1,j1], [i2,j2], [i3,j3] respectively
        p_list: [p1,p2,p3,...] measures with prob p1,p2,p3, respectively
        '''
        if Born:
            assert len(site_list)== len(p_list), f'site_list ({len(site_list)}) is not equal to p_list ({len(p_list)})'
            for (i,j),p in zip(site_list,p_list):
                Gamma=self.C_m_history[-1][[i],[j]]
                n_list=get_Born_tri_op(p,Gamma,rng=self.rng)
                self.measure(n_list,[i,j])
        else:
            pass
    
    def measure_list_tri_op_perfect_teleportation(self,site_list,p_list,Born=True,parity_dict=None):
        '''site_list: [[i1,j1],[i2,j2],[i3,j3],...] measures [i1,j1], [i2,j2], [i3,j3] respectively
        p_list: [p1,p2,p3,...] measures with prob p1,p2,p3, respectively
        outcome:  {(i,j): parity}

        '''
        if Born:
            assert len(site_list)== len(p_list), f'site_list ({len(site_list)}) is not equal to p_list ({len(p_list)})'
            for (i,j),p in zip(site_list,p_list):
                Gamma=self.C_m_history[-1][[i],[j]]
                n_list=get_Born_tri_op(p,Gamma,rng=self.rng)
                self.measure(n_list,[i,j])
                if n_list[0] == [-1,0,0] or [1,0,0]:
                    # P_+ or P_-
                    other_leg=find_other_leg(parity_dict, (i,j))
                    if len(other_leg)==1:
                        self.measure([0,-1,0], [other_leg])
                    update_dictionary(parity_dict,(i,j),p=-n_list[0][0])
                else:
                    # unitary
                    update_dictionary(parity_dict,(i,j),p=None)
        else:
            pass


    def mutual_information_cross_ratio(self,ratio=[1,4],unitcell=1):
        """unitcell=1: shift each fermionic site
        unitcell=2: shift "2-atom" unit cell
        """
        
        x=np.array([0,self.L//ratio[1]*ratio[0],self.L//2,self.L//2+self.L//ratio[1]*ratio[0]])
        # x=np.array([0,self.L//8,self.L//2,self.L//8*5])
        MI=[]
        subA=np.arange(x[0],x[1])
        subB=np.arange(x[2],x[3])
        for shift in np.arange(0,self.L//2,unitcell):
            MI.append(self.mutual_information_m((subA+shift)%self.L, (subB+shift)%self.L))
        return np.mean(MI)
        # return MI

    def entanglement_contour(self,subregion,fermion=False):
        # c_A=self.c_subregion_m(subregion)
        c_A=self.c_subregion_m(subregion)+1e-18j
        C_f=(np.eye(c_A.shape[0])+1j*c_A)/2
        f,_=la.funm(C_f,lambda x: -x*np.log(x),disp=False)
        if fermion:
            return np.diag(f).real.reshape((-1,2)).sum(axis=1)
        else:
            return np.diag(f).real

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

    def von_Neumann_entropy_m_self_average(self,Gamma=None,unitcell=1):
        subregion=np.arange(0,self.L//2)
        EE=[]
        for shift in np.arange(0,self.L//2,unitcell):
            EE.append(self.von_Neumann_entropy_m((subregion+shift)%self.L,Gamma))
        return np.mean(EE)


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
            return np.int_(sorted(np.concatenate([n*subregion+i for i in range(0,n,k)])))
        else:
            return np.int_(sorted(np.concatenate([n*subregion+i for i in range(n)])))
        
def get_random_tri_op(p,num,rng=None):
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    n1= (sign<p/2)*(-1)+(sign>1-p/2)
    # n2,n3=get_inplane(n1, num,rng=rng)
    n2,n3=get_inplane_norm(n1, num,rng=rng,sigma=np.pi/10)
    return np.c_[n1,n2,n3]

def get_Born_tri_op(p,Gamma,rng=None):
    num=Gamma.shape[0]
    rng=np.random.default_rng(rng)
    sign=rng.random(size=num)
    n1= (sign<p*(1+Gamma)/2)*(-1)+(sign>p*(1+Gamma)/2+1-p)
    n2,n3=get_inplane(n1, num,rng=rng)
    return np.c_[n1,n2,n3]

def get_Born_class_AIII(A,Gamma,class_A=False,rng=None):
    rng=np.random.default_rng(rng)
    prob={(s1,s2): Gamma[0,1]*(-s1-s2)/8*A + Gamma[0,3]*(-s1+s2)/8*A + Gamma[1,2]*(s1-s2)/8*A + Gamma[2,3]*(-s1-s2)/8*A - (-Gamma[0,1]*Gamma[2,3]+Gamma[0,2]*Gamma[1,3]-Gamma[0,3]*Gamma[1,2])*s1*s2*A**2/4+1/4 for s1 in [-1,1] for s2 in [-1,1]}
    for key,val in prob.items():
        assert val>-1e-9, f'{key} < 0 = {val}, {prob}'
        assert val<1+1e-9, f'{key} > 1 = {val}'
        if prob[key]>1 or prob[key]<0:
            prob[key]=np.clip(val,0.,1.)
    if not class_A:
        kind=rng.choice(list(prob.keys()),p=list(prob.values()))
    else:
        post_selected_outcome=[(-1,1),(1,-1)]
        norm= sum(prob[i] for i in post_selected_outcome)
        kind=rng.choice(post_selected_outcome,p=[prob[i]/norm for i in post_selected_outcome])
    theta=rng.uniform(-np.pi,np.pi,size=2)
    return tuple(kind), theta[0],theta[1]

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

def get_inplane_norm(n1,num,rng=None,sigma=np.pi/4,mu=0):
    r=np.sqrt(1-n1**2)
    rng=np.random.default_rng(rng)
    phi=rng.normal(loc=mu,scale=sigma,size=num)
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
    """rescale a range [x0,x1] to [y0,y1] using linear map"""
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
def P_contraction(m,proj_list,ix,ix_bar,combine=True):
    ix,ix_bar=list(ix),list(ix_bar)

    if combine:
        proj=np.zeros((4*len(proj_list),4*len(proj_list)))
        # change index from (in_1, in_2, out_1, out_2) (in_3, in_4, out_3, out_4)
        # to (in_1 , in_2, in_3, in_4, out_1, out_2, out_3, out_4)
        for i,p in enumerate(proj_list):
            proj[np.ix_([2*i,2*i+1,2*i+2*len(proj_list),2*i+2*len(proj_list)+1],[2*i,2*i+1,2*i+2*len(proj_list),2*i+2*len(proj_list)+1])]=p
    else:
        assert len(proj_list)==1, 'len of proj_list is not 1'
        proj=proj_list[0]

    # if m.shape[0]==0:
    #     return proj
        
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
        try:
            Psi=mat1+mat2@(nla.solve(mat3,mat2.T))
        except:
        # Psi=mat1+mat2@nla.inv(mat3)@mat2.T
            Psi=mat1+mat2@(la.lstsq(mat3,mat2.T)[0])
    else:
        # print('mat2 is singular')
        Psi=mat1
    
    Psi_mat=np.zeros_like(Psi)
    Psi_mat[np.ix_(ix_bar,ix_bar)]=Psi[:len(ix_bar),:len(ix_bar)]
    Psi_mat[np.ix_(ix,ix)]=Psi[-len(ix):,-len(ix):]
    Psi_mat[np.ix_(ix_bar,ix)]=Psi[:len(ix_bar),-len(ix):]
    Psi_mat[np.ix_(ix,ix_bar)]=Psi[-len(ix):,:len(ix_bar)]
    Psi=Psi_mat

    Psi=(Psi-Psi.T)/2

    # orthogonalize Psi, see App. B2 in PhysRevB.106.134206
    if np.abs(np.diag(Psi@Psi)+1).max()>1e-10:
        Psi=purify(Psi)
        Psi=(Psi-Psi.T)/2
    return (Psi)

def P_contraction_2(Gamma,Upsilon,ix,ix_bar,Gamma_like=None,reset_Gamma_like=True):
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
    Upsilon_LL=Upsilon[:len(ix),:len(ix)]
    Upsilon_RR=Upsilon[len(ix):,len(ix):]
    Upsilon_RL=Upsilon[len(ix):,:len(ix)]
    eye=np.eye(len(ix))
    try: 
        C=nla.inv(Gamma_RR@Upsilon_LL+eye)
    except:
        raise ValueError("the contraction will lead to a vanishing state")
    A=Upsilon_LL@C
    D=Gamma_RR@C.T
    tmp=Gamma_LR@A@Gamma_LR.T
    if Gamma_like is None:
        Gamma_like=np.zeros_like(Gamma)
    if reset_Gamma_like:
        Gamma_like[:,:]=0
    Gamma_like[np.ix_(ix_bar,ix_bar)]=tmp
    Gamma+=Gamma_like
    Gamma[np.ix_(ix,ix_bar)]=Upsilon_RL@C@Gamma_LR.T
    Gamma[np.ix_(ix,ix)]=Upsilon_RR+Upsilon_RL@D@Upsilon_RL.T
    Gamma[np.ix_(ix_bar,ix)]=-Gamma[np.ix_(ix,ix_bar)].T
    # why is it neccessary?
    # Gamma-=Gamma.T
    # Gamma/=2

    if np.abs(np.einsum(Gamma,[0,1],Gamma,[1,0],[0])+1).max()>1e-10:
        Gamma[:,:]=purify(Gamma)
        Gamma-=Gamma.T
        Gamma/=2


def interpolation(x1,x2,l0,h0,L,k=1,sign=1):
    x=np.arange(L)
    h=h0/2
    l=l0-h0/2
    return (h-l)/2*(np.tanh((x-x1)*k)+1)+l-sign*(h-l)/2*(np.tanh((x-x2)*k)+1)+h

def interpolation2(x0_list,l_list,L,k=1):
    """x0 for domain wall position
    l0 for each domain wall hight
    """
    x=np.arange(L)
    # h=h0/2
    # l=l0-h0/2
    s=0
    for x0,l in zip(x0_list,l_list):
        s+=(l)*(np.tanh((x-x0)*k)+1)/2
    return s

def purify(A):
    # purify A, see App. B2 in PhysRevB.106.134206
    val,vec=np.linalg.eigh(A/1j)
    val[val<0]=-1
    val[val>0]=1
    return -(vec@np.diag(val)@vec.conj().T).imag
def purify_O(A,A_D):
    O,_,_=block_diagonalize(A)
    return O.T@A_D@O

def block_diagonalize(A,thres=1e-10):
    '''A is an anti symmetry matrix for covariance matrix
    block diagonalize is to find a real othorgonal matrix such that OAO^T=A_D, where A_D=\oplus a_k \omega, where \omega = [0,1;-1,0]

    See : arxiv:0902.1502 App B for more details
    '''
    assert np.abs(A.imag).max()<1e-10, f'A is not a real matrix {np.abs(A.imag).max()}'
    A=A.real
    assert np.abs(A+A.T).max()<1e-10, f'A is not antisymmetric'

    val,vec=np.linalg.eigh(A/1j)
    val_arg=val.argsort()
    val=val[val_arg[:A.shape[0]//2]]
    vec=vec[:,val_arg[:A.shape[0]//2]]
    perm_list=permutations(range(vec.shape[1]))
    for perm in perm_list:
        vec=vec[:,perm]
        diag_element=np.array([vec[2*x,x] for x in range(vec.shape[1])])
        if np.all(diag_element !=0):
            break

    # diag_element=np.array([vec[2*x,x] for x in range(vec.shape[1])])
    phase_factor=diag_element.conj()/np.abs(diag_element)
    # phase_factor[np.isinf(phase_factor)|np.isnan(phase_factor)]=1
    vec=phase_factor.reshape((1,-1))*vec
    vec_conj=vec.conj()
    U=np.zeros(A.shape,dtype=complex)
    U[:,1::2]=vec_conj
    U[:,::2]=vec
    G=lambda l:np.kron(np.eye(l),np.array([[1,1],[-1j,1j]])/np.sqrt(2))
    O=G(A.shape[0]//2)@U.T.conj()
    assert np.abs(O.imag).max()<1e-10, f'O is not a real matrix {np.abs(O.imag).max()}'
    O=O.real
    A_D=O@A@O.T
    return O,U,A_D

def fidelity(A,B):
    assert A.shape[0]==B.shape[0], f'A {A.shape[0]} has different dim than B{B.shape[0]}'
    L=A.shape[0]//2
    identity=np.eye(A.shape[0])
    id_AB=(identity-A@B)
    if np.linalg.det(id_AB)!=0:
        G_tilde=(A+B)@np.linalg.inv(id_AB)
        # G_tilde=np.linalg.solve((A+B).T,id_AB.T).T    # Right divide
    else:
        G_tilde=(A+B)@np.linalg.inv(id_AB+1e-10*identity).real
        # G_tilde=np.linalg.solve((A+B).T,id_AB.T+1e-8j*np.eye(A.shape[0])).T.real    # Right divide

    sqrt_G_tilde=scipy.linalg.funm(identity+G_tilde@G_tilde,np.sqrt)
    return 2**(-L/2)*np.linalg.det(identity-A@B)**(1/4)*np.linalg.det(identity+sqrt_G_tilde)**(1/4)

    
def update_dictionary(parity_dict,ij,p):
    i,j=ij
    remove (i,anything) and (anything,j) in parity_dict
    if p is not None:
        assert abs(p)==1 , f'p should be 1 or -1 {p}'
        praity_dict[(i,j)]=p

def find_other_leg(parity_dict,ij):
    i,j=ij
    other_leg_list=[]
    for x in parity_dict.keys():
        if i in x or j in x:
            other_leg_list.append(x)
    return other_leg_list
# %%
