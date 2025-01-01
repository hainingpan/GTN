import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import scipy.sparse as sp
from copy import copy
from utils import get_O, circle, get_Born_single_mode, op_single_mode, P_contraction_2, op_fSWAP, get_Born_tri_op, kraus

# Generate 2d lattice model
class GTN2:
    def __init__(self,Lx,Ly,history=True,seed=None,random_init=False,random_U1=False,bcx=1,bcy=1,orbit=1,layer=1,replica=1,nshell=1,sparse=False):
        self.Lx= Lx # complex fermion sites
        self.Ly=Ly # complex fermion sites
        self.L = Lx* Ly*orbit # (Lx,Ly) in complex fermion sites
        self.orbit=orbit # number of orbitals
        self.layer=layer # number of layers
        self.replica=replica # number of replicas, for the reference sites
        self.history = history
        self.sparse= sparse
        self.random_init = random_init
        self.random_U1 = random_U1
        self.rng=np.random.default_rng(seed)
        self.C_m=self.correlation_matrix()
        self.Gamma_like=self.zeros_like(self.C_m)
        self.C_m_history=[self.C_m.copy()]
        self.n_history=[]
        self.i_history=[]
        self.p_history=[]
        self.seed = seed
        self.bcx = bcx # boundary condition in x direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.bcy = bcy # boundary condition in y direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.full_ix=set(range(self.C_m.shape[0]))
        self.nshell = nshell

    def kron(self,*args,**kwargs):
        if self.sparse:
            return sp.kron(*args,**kwargs).tocsr()
        else:
            return np.kron(*args,**kwargs)
    def zeros_like(self,A):
        if self.sparse:
            return sp.csr_matrix(A.shape)
        else:
            return np.zeros_like(A)

    def set(self,ij_list,n,Gamma=None):
        """ij_list: [[i,j],...]
        n: [1,-1,...]
        simply set all Gamma[i,j]=n, Gamma[j,i]=-n
        """
        if Gamma is None:
            Gamma=self.C_m
        for ij,n in zip(ij_list,n):
            i,j=ij
            Gamma[i,j]=n
            Gamma[j,i]=-n

    def correlation_matrix(self):
        L_complex_f=self.replica*self.layer*self.L
        Omega=np.array([[0,1.],[-1.,0]])
        
        if self.replica==2:
            # the reference sites are entangled with their own site indices
            Omega_diag=self.kron(Omega,np.eye(L_complex_f))
        else:
            Omega_diag=self.kron(np.eye(L_complex_f),Omega)
        if self.random_init:
            if self.random_U1:
                # random with U1
                i_list = self.rng.random(L_complex_f)
                i_list = 2*np.arange(L_complex_f)[i_list<0.5]
                j_list = i_list+1
                ij_list=np.array([i_list,j_list]).T
                self.set(ij_list=ij_list,n=[-1]*i_list.shape[0],Gamma=Omega_diag)
                Gamma=Omega_diag
            else:
                # random without U1, in this scenario, the sparse matrix is not optimal
                O=get_O(self.rng,2*L_complex_f)
                Gamma=O@Omega_diag@O.T
        else:
            Gamma=Omega_diag
        return (Gamma-Gamma.T)/2
    
    def measure_tri_op(self,i,j,p,Born=True,):
        '''measure the parity on i,j (Majorana sites) with probability p
        '''
        if Born:
            Gamma=self.C_m[i,j]
            n=get_Born_tri_op(p,Gamma,rng=self.rng)
            self.measure(n[0],[i,j])
        else:
            pass
    def randomize(self,legs):
        """ legs is the majorana site index, 
        simply randomize the parity"""
        n_list=get_Born_tri_op(p=0,Gamma=np.array([0]),rng=self.rng,alpha=1)
        self.measure(n_list[0],ix=legs)
        return n_list[0][0]
        
    def measure(self,n,ix):
        ''' Majorana site index for ix, 
        n should be a scalar'''

        Psi=self.C_m
        proj=kraus(n)
        ix_bar=np.array(list(self.full_ix-set(ix)))

        # Psi=P_contraction(m,proj,ix,ix_bar)
        P_contraction_2(Psi,proj,ix,ix_bar,self.Gamma_like,reset_Gamma_like=False)

        # assert np.abs(np.trace(Psi))<1e-5, "Not trace zero {:e}".format(np.trace(Psi))
        if self.history:
            self.C_m_history.append(Psi.copy())
            self.n_history.append(n)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            self.C_m_history=[Psi]
            self.n_history=[n]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]



    def measure_weak(self,A,ix,kind):
        """measure exp(-beta c^dag c), A = tanh(beta), sqrt(1-A^2) = 1/cosh(beta)"""
        Psi=self.C_m
        ix_bar=np.array(list(self.full_ix-set(ix)))
        if kind == 'onsite':
            proj=op_weak_onsite(A)
        elif kind == 'nn_x':
            proj=op_weak_nn_x(A)
        elif kind == 'nn_y':
            proj=op_weak_nn_y(A)
        P_contraction_2(Psi,proj,ix,ix_bar,self.Gamma_like,reset_Gamma_like=False)
        if self.history:
            self.C_m_history.append(Psi.copy())
            self.n_history.append([A])
            self.i_history.append(ix)
        else:
            self.n_history=[A]
            self.i_history=[ix]
    def measure_single_mode_Born(self,legs,mode):
        """measure the single mode with mode = (wf, n), wavefunction and occupation number 
        """
        Gamma = self.C_m[np.ix_(legs,legs)]
        n = get_Born_single_mode(Gamma=Gamma,mode=mode,rng=self.rng)
        self.measure_single_mode_force(kind=(mode,n),ix=legs)
        return (mode,n)

    def measure_single_mode_force(self,kind,ix,):
        ''' Majorana site index for ix'''
        assert len(ix)==len(kind[0])*2, 'len of ix should be 2*len(kind[0])'
        Psi=self.C_m
        ix_bar=np.array(list(self.full_ix-set(ix)))
        kind = (tuple(kind[0]),kind[1])
        proj=op_single_mode(kind)
        P_contraction_2(Psi,proj,ix,ix_bar,self.Gamma_like,reset_Gamma_like=True)
        if self.history:
            self.C_m_history.append(Psi.copy())
            self.n_history.append(kind)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            # self.C_m_history=[Psi]
            self.n_history=[kind]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]
    
    
    def fSWAP(self,ix,state1=None,state2=None):
        """ Majorana site index for ix ,exp(i*pi*c_-^dag c_-)"""
        Psi=self.C_m
        ix_bar=np.array(list(self.full_ix-set(ix)))
        op=op_fSWAP(state1,state2)
        P_contraction_2(Psi,op,ix,ix_bar,self.Gamma_like,reset_Gamma_like=True)
        if self.history:
            self.C_m_history.append(Psi.copy())
            self.n_history.append([state1,state2])
            self.i_history.append(ix)
        else:
            self.n_history=[state1,state2]
            self.i_history=[ix]

    def measure_feedback(self,ij,mu=None,feedback=True,region=None):
        """ix is the 2 fermionic site index
        this can be used to incorporate feedback"""
        i,j=ij
        if mu is None:
            mu = list(self.a_i.keys())[0]
        legs_t_lower,wf_lower=self.generate_ij_wf(i,j,self.a_i[mu],self.b_i[mu],self.bcx,self.bcy,region=region)
        legs_t_upper,wf_upper=self.generate_ij_wf(i,j,self.A_i[mu],self.B_i[mu],self.bcx,self.bcy,region=region)

        legs_bA = [self.linearize_idx(i=i,j=j,orbit_idx=0,majorana=majorana)+2*self.L for majorana in range(2)]
        legs_bB = [self.linearize_idx(i=i,j=j,orbit_idx=1,majorana=majorana)+2*self.L for majorana in range(2)]
        # print(legs_bA,legs_bB)
        # print(legs_t_lower)

        # fill lower band
        mode_m,n_m=self.measure_single_mode_Born(legs_t_lower,mode=wf_lower)
        if n_m ==1:
            # this is good
            pass
        elif n_m ==0:
            self.fSWAP(legs_t_lower+legs_bA,state1 = wf_lower, state2=(1,))
            self.fSWAP(legs_t_lower+legs_bB,state1 = wf_lower, state2=(1,))
            # _, n_bA = self.measure_single_mode_Born(legs_bA,mode=[1])
            # if n_bA == 1:
            #     # A bottom occupied, fill to top layer
            #     self.fSWAP(legs_t_lower+legs_bA,state1 = wf_lower, state2=[1])
            # elif n_bA == 0:
            #     # A bottom empty, measure B bottom
            #     _, n_bB = self.measure_single_mode_Born(legs_bB,mode=[1])
            #     if n_bB ==1:
            #         # B bottom occupied, fill to top layer
            #         self.fSWAP(legs_t_lower+legs_bB,state1 = wf_lower, state2=[1])
            #     elif n_bB ==0:
            #         # B bottom also empty
            #         pass 
            #     else:
            #         raise ValueError('Protocol failed')

        # deplete upper band
        mode_p,n_p=self.measure_single_mode_Born(legs_t_upper,mode=wf_upper)
        if n_p ==0:
            # this is good
            pass
        elif n_p == 1:
            self.fSWAP(legs_t_upper+legs_bA,state1 = wf_upper, state2=(1,))
            self.fSWAP(legs_t_upper+legs_bB,state1 = wf_upper, state2=(1,))
            # _, n_bA = self.measure_single_mode_Born(legs_bA,mode=[1])
            # if n_bA == 0:
            #     # A bottom empty, deplete upper band
            #     self.fSWAP(legs_t_upper+legs_bA,state1 = wf_upper, state2=[1])
            # elif n_bA == 1:
            #     # A bottom occupied, measure B bottom
            #     _, n_bB = self.measure_single_mode_Born(legs_bB,mode=[1])
            #     if n_bB == 0:
            #         # B bottom empty
            #         self.fSWAP(legs_t_upper+legs_bB,state1 = wf_upper, state2=[1])
            #     elif n_bB ==1:
            #         # B bottom also occupied
            #         pass
            #     else:
            #         raise ValueError('Protocol failed')


    
    def measure_line_tri_op(self,p_list,pos,Born=True,even=True):
        """ apply along a specific line, e.g., (0,-1), mean all first index being 0 will be applied. (-1,1) means all second index being 1 will be measured
        even means it is "intra-unit cell" gamma
        odd means it is "inter-unit cell" gamma"""
        if pos[0]==-1 and pos[1]>-1:
            ix=[(i,pos[1]) for i in range(self.Lx)]
        elif pos[1]==-1 and pos[0]>-1:
            ix=[(pos[0],i) for i in range(self.Ly)]
        else:
            raise ValueError("pos should have only one `-1` to indicate the direction")
        if even:
            proj_range_1 = [self.linearize_idx(*idx,majorana=0) for idx in ix]
            proj_range_2 = [self.linearize_idx(*idx,majorana=1) for idx in ix]
        else:
            ix_ = ix[1:]+[ix[0]]
            if (pos[0]==-1 and pos[1]>-1 and self.bcy==0) or (pos[1]==-1 and pos[0]>-1 and self.bcx==0):
                ix = ix[:-1]
                ix_ = ix_[:-1]
            proj_range_1 = [self.linearize_idx(*idx,majorana=1) for idx in ix]
            proj_range_2 = [self.linearize_idx(*idx,majorana=0) for idx in ix_]
        # return proj_range_1,proj_range_2
        if isinstance(p_list,int) or isinstance(p_list,float):
            p_list=np.array([p_list]*len(proj_range_1))
        if self.history:
            self.p_history.append(p_list)
        else:
            self.p_history=[p_list]

        if Born:
            for i,j,p in zip(proj_range_1,proj_range_2,p_list):
                Gamma=self.C_m[i,j]
                n=get_Born_tri_op(p,Gamma,rng=self.rng)
                self.measure(n[0],[i,j])
        else:
            pass
    
    def measure_tri_sites(self,A,Amu,ij):
        """force measurement on a single "tri-site" (i,i+x,i+y)"""
        i,j = ij
        self.measure_weak(-Amu,ix=[self.linearize_idx(*ij,orbit_idx=0,majorana=idx) for idx in range(2)],kind='onsite')   # exp(-beta * c_iL^dag c_iL + h.c.)
        self.measure_weak(Amu,ix=[self.linearize_idx(*ij,orbit_idx=1,majorana=idx) for idx in range(2)],kind='onsite')   # exp(beta * c_iR^dag c_iR + h.c.)

        if self.bcx==1 or (i+1)<self.Lx:
            ij_x = [(i+1)%self.Lx,j]
            self.measure_weak(-A,ix=[self.linearize_idx(*ij,orbit_idx=0,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_x,orbit_idx=0,majorana=idx) for idx in range(2)],kind='nn_x')    # exp(-beta * c_iL^dag c_i+x,L + h.c.)
            self.measure_weak(A,ix=[self.linearize_idx(*ij,orbit_idx=1,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_x,orbit_idx=1,majorana=idx) for idx in range(2)],kind='nn_x')   # exp(beta * c_iR^dag c_i+x,R + h.c.)
            self.measure_weak(A,ix=[self.linearize_idx(*ij,orbit_idx=0,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_x,orbit_idx=1,majorana=idx) for idx in range(2)],kind='nn_x')   # exp(beta * c_iL^dag c_i+x,R + h.c.)
            self.measure_weak(-A,ix=[self.linearize_idx(*ij,orbit_idx=1,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_x,orbit_idx=0,majorana=idx) for idx in range(2)],kind='nn_x')   # exp(-beta * c_iR^dag c_i+x,L + h.c.)

        if self.bcy==1 or (j+1)<self.Ly:
            ij_y = [i,(j+1)%self.Ly]
            self.measure_weak(A,ix=[self.linearize_idx(*ij,orbit_idx=0,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_y,orbit_idx=0,majorana=idx) for idx in range(2)],kind='nn_x')    # exp(beta * (-i*c_iL^dag c_i+y,L + h.c.)
            self.measure_weak(-A,ix=[self.linearize_idx(*ij,orbit_idx=1,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_y,orbit_idx=1,majorana=idx) for idx in range(2)],kind='nn_x')    # exp(-beta * (-i*c_iL^dag c_i+y,L + h.c.)

            self.measure_weak(A,ix=[self.linearize_idx(*ij,orbit_idx=0,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_y,orbit_idx=1,majorana=idx) for idx in range(2)],kind='nn_y')    # exp(beta * (-i*c_iL^dag c_i+y,L + h.c.)
            self.measure_weak(A,ix=[self.linearize_idx(*ij,orbit_idx=1,majorana=idx) for idx in range(2)]+[self.linearize_idx(*ij_y,orbit_idx=0,majorana=idx) for idx in range(2)],kind='nn_y')    # exp(beta * (-i*c_iL^dag c_i+y,L + h.c.)

    # def op_Wannier(self,n,lower):
    #     """ project to "n" occupancy of "wf" mode"""
    #     if lower:
    #         wf = np.stack([(a,b) for a,b in zip(self.a_i.values(),self.b_i.values())]).flatten()
    #     else:
    #         wf = np.stack([(a,b) for a,b in zip(self.A_i.values(),self.B_i.values())]).flatten()

    #     # u=form_basis(wf)
    #     return Gamma_n1(tuple(wf),n)


    # def measure_Wannier(self,ij,n,lower):
    #     """measure the single mode centered at ij, project to the Wannier state"""
    #     Psi=self.C_m

    #     i,j = ij
    #     if lower:
    #         ij_list = [((i+di)%self.Lx,(j+dj)%self.Ly) for di,dj in self.a_i.keys()]
    #     else:
    #         ij_list = [((i+di)%self.Lx,(j+dj)%self.Ly) for di,dj in self.A_i.keys()]
    #     ix= [self.linearize_idx(*ij,orbit_idx=orbit_idx,majorana=idx) for ij in ij_list for orbit_idx in range(2) for idx in range(2)]
    #     ix_bar=np.array(list(self.full_ix-set(ix)))
    #     proj = self.op_Wannier(n=n,lower=lower)
    #     P_contraction_2(Psi,proj,ix,ix_bar,self.Gamma_like,reset_Gamma_like=False)
    #     if self.history:
    #         self.C_m_history.append(Psi.copy())
    #         # self.n_history.append([r])
    #         self.i_history.append(ix)
    #     else:
    #         # self.n_history=[r]
    #         self.i_history=[ix]

    # def measure_Wannier_Born(self,ij,lower):
    #     i,j = ij
    #     if lower:
    #         ij_list = [((i+di)%self.Lx,(j+dj)%self.Ly) for di,dj in self.a_i.keys()]
    #         wf = np.stack([(a,b) for a,b in zip(self.a_i.values(),self.b_i.values())]).flatten()
    #     else:
    #         ij_list = [((i+di)%self.Lx,(j+dj)%self.Ly) for di,dj in self.A_i.keys()]
    #         wf = np.stack([(a,b) for a,b in zip(self.A_i.values(),self.B_i.values())]).flatten()
    #     ix= [self.linearize_idx(*ij,orbit_idx=orbit_idx,majorana=idx) for ij in ij_list for orbit_idx in range(2) for idx in range(2)]
    #     ix_bar=np.array(list(self.full_ix-set(ix)))
    #     Gamma = self.C_m[np.ix_(ix,ix)]
            
    #     n = get_Born_single_mode(Gamma=Gamma,mode=wf,rng=self.rng)
    #     self.measure_Wannier(ij,n,lower)
    #     return wf,n

    def linearize_idx(self,i,j,majorana=0,orbit_idx=0,layer=0,replica=0):
        return np.ravel_multi_index((replica,layer,i,j,orbit_idx,majorana),(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    def linearize_idx_span(self,ilist,jlist,layer = 0,replica=0,shape_func=lambda i,j: True, ):
        multi_index=np.array([(replica,layer,i%self.Lx,j%self.Ly,orbit,maj) for i in ilist for j in jlist for orbit in range(self.orbit) for maj in range(2) if shape_func(i%self.Lx,j%self.Ly)])
        return np.ravel_multi_index(multi_index.T,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    def delinearize_idx(self,idx):
        return np.unravel_index(idx,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))


    def von_Neumann_entropy_m(self,subregion,Gamma=None,fermion_idx=True):
        c_A=self.c_subregion_m(subregion,Gamma,fermion_idx=fermion_idx)
        val=nla.eigvalsh(1j*c_A)
        # self.val_sh=val
        val=np.sort(val)
        val=(1-val)/2+1e-18j   #\lambda=(1-\xi)/2
        return np.real(-np.sum(val*np.log(val))-np.sum((1-val)*np.log(1-val)))/2

    def entanglement_contour(self,subregion,fermion=False, Gamma=None, fermion_idx=True):
        # c_A=self.c_subregion_m(subregion)
        c_A=self.c_subregion_m(subregion,Gamma,fermion_idx=fermion_idx)
        C_f=(np.eye(c_A.shape[0])+1j*c_A)/2
        f,_=la.funm(C_f,lambda x: -x*np.log(x),disp=False)
        if fermion:
            return np.diag(f).real.reshape((-1,2)).sum(axis=1)
        else:
            return np.diag(f).real
    def c_subregion_m(self,subregion,Gamma=None,fermion_idx=True):
        if Gamma is None:
            Gamma=self.C_m
        if fermion_idx:
            subregion=self.linearize_index(subregion,2)
        return Gamma[np.ix_(subregion,subregion)]
    def generate_tripartite_circle(self,radius_factor=(2.6,2.6)):
        A_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,center=[self.Lx/2,self.Ly/2],radius=[self.Lx/radius_factor[0],self.Ly/radius_factor[1]],angle=[0,np.pi/3*2]))
        B_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,center=[self.Lx/2,self.Ly/2],radius=[self.Lx/radius_factor[0],self.Ly/radius_factor[1]],angle=[np.pi/3*2,np.pi/3*4]))
        C_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,center=[self.Lx/2,self.Ly/2],radius=[self.Lx/radius_factor[0],self.Ly/radius_factor[1]],angle=[np.pi/3*4,np.pi/3*6]))
        return A_idx_0,B_idx_0,C_idx_0
    
    def generate_ij_wf(self,i,j,a_i,b_i,bcx,bcy,region=None):
        """generate ij_list from a local mode a_i;
        assume a_i, and b_i have the same keys"""
        ij_list = []
        wf = []
        for di,dj in a_i.keys():
            i1,j1=(i+di),(j+dj)
            if bcx==1:
                i1=i1%self.Lx
            elif bcx==0:
                if i1<0 or i1>=self.Lx:
                    continue
            if bcy==1:
                j1=j1%self.Ly
            elif bcy==0:
                if j1<0 or j1>=self.Ly:
                    continue
            if region is not None:
                if (i1,j1) not in region:
                    continue
            ij_list.append((i1,j1))
            wf.append(a_i[di,dj])
            wf.append(b_i[di,dj])
        legs=[self.linearize_idx(*ij,orbit_idx=orbit_idx,majorana=idx) for ij in ij_list for orbit_idx in range(2) for idx in range(2)]
        return legs, tuple(wf)


def amplitude(nshell,nkx=500,nky=500,tau=[0,1],mu=1,geometry = 'square', lower=True, C=1):
    """if gemoetry is square, then the shape is [i-nshell,i+nshell]x[j-nshell,j+nshell]"""
    
    kx = np.linspace(-np.pi,np.pi,nkx)
    ky = np.linspace(-np.pi,np.pi,nky)
    KX,KY = np.meshgrid(kx,ky, indexing='ij')
    offdiag=(np.sin(KX)-1j*np.sin(KY))**C
    dx = offdiag.real
    dy = -offdiag.imag
    dz = mu-np.cos(KX)-np.cos(KY)
    E = np.sqrt(dx**2+dy**2+dz**2)
    cos_theta = dz/E
    sin_theta_exp_iphi = (dx+1j*dy)/E
    tau = np.array(tau)/np.linalg.norm(tau)
    if lower:
        ak = (1-cos_theta)/2*tau[0] - 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = - 1/2*sin_theta_exp_iphi*tau[0] + (1+cos_theta)/2*tau[1]
    else:
        ak = (1+cos_theta)/2*tau[0] + 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = 1/2*sin_theta_exp_iphi*tau[0] + (1-cos_theta)/2*tau[1]

    # ak=-1/2*(dx-1j*dy)/E
    # bk=1/2+dz/(2*E)

    def a_nint(i,j):
        a_int=ak * np.exp(1j * (KX * i + KY*j))
        return np.trapz(np.trapz(a_int,kx),ky)/(2*np.pi)**2
    def b_nint(i,j):
        b_int=bk * np.exp(1j * (KX * i + KY*j))
        return np.trapz(np.trapz(b_int,kx),ky)/(2*np.pi)**2

    if geometry == 'square':
        i_list = np.arange(-nshell,nshell+1)
        j_list = np.arange(-nshell,nshell+1)
        ij_list = [(i,j) for i in i_list for j in j_list]
    elif geometry == 'diamond':
        ij_list=[(i,j) for i in range(-nshell,nshell+1) for j in range(-nshell+abs(i),nshell+1 -abs(i))]
    a_i = {(i,j):a_nint(i,j) for i,j in ij_list}
    b_i = {(i,j):b_nint(i,j) for i,j in ij_list}
    return a_i,b_i


def amplitude_fft(nkx=5000,nky=5000,tau=[0,1],mu=1, lower=True, C=1):
    """if gemoetry is square, then the shape is [i-nshell,i+nshell]x[j-nshell,j+nshell]"""
    
    kx = np.linspace(-np.pi,np.pi,nkx)
    ky = np.linspace(-np.pi,np.pi,nky)
    KX,KY = np.meshgrid(kx,ky, indexing='ij')
    offdiag=(np.sin(KX)-1j*np.sin(KY))**C
    dx = offdiag.real
    dy = -offdiag.imag
    dz = mu-np.cos(KX)-np.cos(KY)
    E = np.sqrt(dx**2+dy**2+dz**2)
    cos_theta = dz/E
    sin_theta_exp_iphi = (dx+1j*dy)/E
    tau = np.array(tau)/np.linalg.norm(tau)
    if lower:
        ak = (1-cos_theta)/2*tau[0] - 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = - 1/2*sin_theta_exp_iphi*tau[0] + (1+cos_theta)/2*tau[1]
    else:
        ak = (1+cos_theta)/2*tau[0] + 1/2*sin_theta_exp_iphi.conj()*tau[1]
        bk = 1/2*sin_theta_exp_iphi*tau[0] + (1-cos_theta)/2*tau[1]

    a_i = np.fft.fft2(ak)/(nkx*nky)
    b_i = np.fft.fft2(bk)/(nkx*nky)
    return a_i,b_i


