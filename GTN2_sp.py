import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import scipy.sparse as sp
from utils import get_O, get_Born_single_mode, op_fSWAP, op_single_mode
from utils_sp import P_contraction_sp
class GTN2_sp:
    def __init__(self,Lx,Ly,history=True,seed=None,random_init=False,random_U1=False,bcx=1,bcy=1,orbit=1,layer=1,replica=1,nshell=1,):
        self.Lx= Lx # complex fermion sites
        self.Ly=Ly # complex fermion sites
        self.L = Lx* Ly*orbit # (Lx,Ly) in complex fermion sites
        self.orbit=orbit # number of orbitals
        self.layer=layer # number of layers
        self.replica=replica # number of replicas, for the reference sites
        self.history = history
        self.random_init = random_init
        self.random_U1 = random_U1
        self.rng=np.random.default_rng(seed)
        self.C_m=self.correlation_matrix()
        # self.Gamma_like=zeros_like(self.C_m)
        self.C_m_history=[self.C_m.copy()]
        self.n_history=[]
        self.i_history=[]
        self.p_history=[]
        self.seed = seed
        self.bcx = bcx # boundary condition in x direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.bcy = bcy # boundary condition in y direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.full_ix=set(range(self.C_m.shape[0]))
        self.nshell = nshell


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
            Omega_diag=sp.kron(Omega,np.eye(L_complex_f),format='csr')
        else:
            Omega_diag=sp.kron(np.eye(L_complex_f),Omega,format='csr')
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
                O=(get_O(self.rng,2*L_complex_f))
                Gamma=sp.csr_matrix(O@Omega_diag@O.T)
        else:
            Gamma=Omega_diag
        return (Gamma-Gamma.T)/2
    
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

        # deplete upper band
        mode_p,n_p=self.measure_single_mode_Born(legs_t_upper,mode=wf_upper)
        if n_p ==0:
            # this is good
            pass
        elif n_p == 1:
            self.fSWAP(legs_t_upper+legs_bA,state1 = wf_upper, state2=(1,))
            self.fSWAP(legs_t_upper+legs_bB,state1 = wf_upper, state2=(1,))

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
        proj=op_single_mode(kind,sparse=True)
        self.C_m=P_contraction_sp(Psi,proj,ix,ix_bar)
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
        op=op_fSWAP(state1,state2,sparse=True)
        self.C_m=P_contraction_sp(Psi,op,ix,ix_bar)
        if self.history:
            self.C_m_history.append(Psi.copy())
            self.n_history.append([state1,state2])
            self.i_history.append(ix)
        else:
            self.n_history=[state1,state2]
            self.i_history=[ix]

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
    
    def linearize_idx(self,i,j,majorana=0,orbit_idx=0,layer=0,replica=0):
        return np.ravel_multi_index((replica,layer,i,j,orbit_idx,majorana),(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    def linearize_idx_span(self,ilist,jlist,layer = 0,replica=0,shape_func=lambda i,j: True, ):
        multi_index=np.array([(replica,layer,i%self.Lx,j%self.Ly,orbit,maj) for i in ilist for j in jlist for orbit in range(self.orbit) for maj in range(2) if shape_func(i%self.Lx,j%self.Ly)])
        return np.ravel_multi_index(multi_index.T,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    def delinearize_idx(self,idx):
        return np.unravel_index(idx,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))


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
