import numpy as np
import numpy.linalg as nla
import torch
import time
from utils_torch import P_contraction_torch, get_O
from utils import op_single_mode, op_fSWAP, circle

class GTN2_torch:
    def __init__(self,Lx,Ly,history=True,seed=None,random_init=False,random_U1=False,bcx=1,bcy=1,orbit=1,layer=1,replica=1,nshell=1,gpu=True,complex128=True,err=1e-8):
        self.Lx= Lx # complex fermion sites
        self.Ly=Ly # complex fermion sites
        self.L = Lx* Ly*orbit # (Lx,Ly) in complex fermion sites
        self.orbit=orbit # number of orbitals
        self.layer=layer # number of layers
        self.replica=replica # number of replicas, for the reference sites
        self.gpu=gpu
        self.device=self._initialize_device()
        self.history = history
        self.random_init = random_init
        self.random_U1 = random_U1
        self.dtype_float=torch.float64 if complex128 else torch.float32
        self.dtype_complex=torch.complex128 if complex128 else torch.complex64
        self.err=torch.tensor(err,device=self.device)
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        self.C_m=self.correlation_matrix()
        self.Gamma_like=torch.zeros_like(self.C_m)
        self.C_m_history=[self.C_m.cpu().clone()]
        self.n_history=[]
        self.i_history=[]
        self.p_history=[]
        self.seed = seed
        self.bcx = bcx # boundary condition in x direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.bcy = bcy # boundary condition in y direction, 0 for open, 1 for periodic, -1 for antiperiodic
        self.full_ix=set(range(self.C_m.shape[0]))
        self.ix_bool=torch.zeros(self.C_m.shape[0],dtype=torch.bool,device=self.device)
        self.nshell = nshell
        self.S0=torch.tensor([[1,1j],[1,-1j]],device=self.device,dtype=self.dtype_complex)/2
    
    def _initialize_device(self):
        """Initialize the device, if `gpu` is True, use GPU, otherwise, use CPU.

        Returns
        -------
        cuda instance
            the name of GPU device

        Raises
        ------
        ValueError
            If GPU is not available, raise error.
        """
        if self.gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                print('Using',device)
                print(f"GPU Model: {gpu_name}")
                return device
            else:
                raise ValueError('CUDA is not available')
        else:
            print('Using cpu')
    
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
        Omega=torch.tensor([[0,1.],[-1.,0]],device=self.device,dtype=self.dtype_float)
        eyeL=torch.eye(L_complex_f,device=self.device,dtype=self.dtype_float)
        if self.replica==2:
            # the reference sites are entangled with their own site indices
            Omega_diag=torch.kron(Omega,eyeL)
        else:
            Omega_diag=torch.kron(eyeL,Omega)
        if self.random_init:
            if self.random_U1:
                # random with U1
                i_list = torch.rand(size=(L_complex_f,), generator=self.rng, device=self.device, dtype=self.dtype_float)
                i_list = 2*torch.arange(L_complex_f,device=self.device)[i_list<0.5]
                j_list = i_list+1
                ij_list=torch.vstack([i_list,j_list]).T
                self.set(ij_list=ij_list,n=[-1]*i_list.shape[0],Gamma=Omega_diag)
                Gamma=Omega_diag
            else:
                # random without U1, in this scenario, the sparse matrix is not optimal
                O=get_O(self.rng,2*L_complex_f,device=self.device,dtype=self.dtype_float)
                Gamma=O@Omega_diag@O.T
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


        # fill lower band
        mode_m,n_m=self.measure_single_mode_Born(legs_t_lower,mode=wf_lower)
        if n_m ==1:
            # this is good
            pass
        elif n_m ==0:
            self.fSWAP(legs_t_lower+legs_bA,state1 = wf_lower, state2=(1,))
            # self.fSWAP(legs_t_lower+legs_bB,state1 = wf_lower, state2=(1,))

        # deplete upper band
        mode_p,n_p=self.measure_single_mode_Born(legs_t_upper,mode=wf_upper)
        if n_p ==0:
            # this is good
            pass
        elif n_p == 1:
            # self.fSWAP(legs_t_upper+legs_bA,state1 = wf_upper, state2=(1,))
            self.fSWAP(legs_t_upper+legs_bB,state1 = wf_upper, state2=(1,))
    
    def order_parameter(self,mu=None,region=None):
        """ the order parameter is defined as sum_{ij} (1-n_- + n_+)/L; such that it is effective the defect density per unit cell"""
        if mu is None:
            mu = list(self.a_i.keys())[0]
        n_lower, n_upper = 0,0
        for i in range(self.Lx):
            for j in range(self.Ly):
                legs_t_lower,wf_lower=self.generate_ij_wf(i,j,self.a_i[mu],self.b_i[mu],self.bcx,self.bcy,region=region)
                legs_t_upper,wf_upper=self.generate_ij_wf(i,j,self.A_i[mu],self.B_i[mu],self.bcx,self.bcy,region=region)
                legs_t_lower=torch.tensor(legs_t_lower,device=self.device)
                legs_t_upper=torch.tensor(legs_t_upper,device=self.device)
                Gamma = self.C_m[legs_t_lower[:,None],legs_t_lower[None,:]]
                n_lower += self.get_Born(Gamma,u=wf_lower)
                Gamma = self.C_m[legs_t_upper[:,None],legs_t_upper[None,:]]
                n_upper += self.get_Born(Gamma,u=wf_upper)
        return 1-(n_lower-n_upper)/(self.Lx*self.Ly)


                

    def measure_single_mode_Born(self,legs,mode):
        """measure the single mode with mode = (wf, n), wavefunction and occupation number 
        """
        legs=torch.tensor(legs,device=self.device)
        Gamma = self.C_m[legs[:,None],legs[None,:]]
        n = self.get_Born_single_mode(Gamma=Gamma,mode=mode,rng=self.rng)
        self.measure_single_mode_force(kind=(mode,n),ix=legs)
        return (mode,n)

    def measure_single_mode_force(self,kind,ix,):
        ''' Majorana site index for ix'''
        assert len(ix)==len(kind[0])*2, 'len of ix should be 2*len(kind[0])'
        Psi=self.C_m
        # ix_bar=torch.tensor(list(self.full_ix-set(ix.tolist())),device=self.device)
        ix_bar=self.complement(ix)
        kind = (tuple(kind[0]),kind[1])
        proj=torch.tensor(op_single_mode(kind),device=self.device,dtype=self.dtype_float)
        P_contraction_torch(Psi,proj,ix,ix_bar,device=self.device,err=self.err,Gamma_like=self.Gamma_like,reset_Gamma_like=True)
        if self.history:
            self.C_m_history.append(Psi.cpu().clone())
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
        # ix_bar=torch.tensor(list(self.full_ix-set(ix)),device=self.device)
        ix_bar=self.complement(ix)
        ix=torch.tensor(ix,device=self.device)
        Psi=self.C_m
        op=torch.tensor(op_fSWAP(state1,state2),device=self.device,dtype=self.dtype_float)
        P_contraction_torch(Psi,op,ix,ix_bar,device=self.device,err=self.err,Gamma_like=self.Gamma_like,reset_Gamma_like=True)
        if self.history:
            self.C_m_history.append(Psi.cpu().clone())
            self.n_history.append([state1,state2])
            self.i_history.append(ix)
        else:
            self.n_history=[state1,state2]
            self.i_history=[ix]

    def randomize(self,legs, scale=1):
        """ legs is the majorana site index, 
        simply randomize the parity"""
        phi=torch.rand((1,),generator=self.rng,device=self.device,dtype=self.dtype_float)*2*np.pi * scale
        n_list=torch.tensor([0,torch.cos(phi),torch.sin(phi)],device=self.device,dtype=self.dtype_float)
        self.measure(n_list,ix=legs)

    def measure(self,n,ix):
        ''' Majorana site index for ix, 
        n should be a scalar'''
        # ix_bar=torch.tensor(list(self.full_ix-set(ix)),device=self.device)
        ix_bar=self.complement(ix)
        ix=torch.tensor(ix,device=self.device)
        Psi=self.C_m
        proj=self.kraus(n)
        P_contraction_torch(Psi,proj,ix,ix_bar,device=self.device,err=self.err,Gamma_like=self.Gamma_like,reset_Gamma_like=True)

        if self.history:
            self.C_m_history.append(Psi.cpu().clone())
            self.n_history.append(n)
            self.i_history.append(ix)
            # self.MI_history.append(self.mutual_information_cross_ratio())
        else:
            # self.C_m_history=[Psi]
            self.n_history=[n]
            self.i_history=[ix]
            # self.MI_history=[self.mutual_information_cross_ratio()]

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

    # def linearize_idx_0(self,i,j,majorana=0,orbit_idx=0,layer=0,replica=0):
    #     if isinstance(i,int):
    #         i=[i]
    #     linear_idx_rev= torch.tensor([majorana,orbit_idx,j,i,layer,replica],device=self.device)
    #     weights= torch.tensor([self.replica,self.layer,self.Lx,self.Ly,self.orbit,2],device=self.device).comprod_(dim=0)
        # return np.ravel_multi_index(,())


    # def linearize_idx(self,index, shape):
    #     out = []
    #     for dim in reversed(shape):
    #         out.append(index % dim)
    #         index = index // dim
    #     return tuple(reversed(out))

    def linearize_idx_span(self,ilist,jlist,layer = 0,replica=0,shape_func=lambda i,j: True, shift=(0,0)):
        multi_index=np.array([(replica,layer,(i+shift[0])%self.Lx,(j+shift[1])%self.Ly,orbit,maj) for i in ilist for j in jlist for orbit in range(self.orbit) for maj in range(2) if shape_func(i%self.Lx,j%self.Ly)])
        return np.ravel_multi_index(multi_index.T,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))

    def delinearize_idx(self,idx):
        return np.unravel_index(idx,(self.replica,self.layer,self.Lx,self.Ly,self.orbit,2))
        
    def get_Born_single_mode(self,Gamma,mode,rng):
        """get the outcome of Born measurement for a single mode, 0 or 1, where mode is sum mode[i] c_i^dag"""
        prob = self.get_Born(Gamma,mode)
        # print(prob)
        if torch.rand((1,),generator=self.rng,device=self.device,dtype=self.dtype_float)< prob:
            return 1
        else:
            return 0

    def get_Born(self,Gamma,u):
        """ get the number density of <V^dag V> where V^dag = sum u_i c_i^dag, C_f is the correlation matrix defined as <c_i^dag c_j>"""
        C_f = self.get_C_f(Gamma)
        u = torch.tensor(u,device=self.device,dtype=self.dtype_complex)
        u/=torch.linalg.norm(u)
        n = u@C_f@u.conj()
        # assert np.abs(n.imag)<1e-10, f'number density is not real {n.imag.max()}'
        return n.real


    def get_C_f(self,Gamma,normal=True):
        """ get the correlation matrix defined as <c_i^dag c_j>"""
        L=Gamma.shape[0]//2
        S = torch.kron(torch.eye(L,device=self.device),self.S0)
        C_f = S@ (torch.eye(2*L,device=self.device)-1j * Gamma) @S.conj().T
        if normal:
            return C_f[::2,::2]
        else:
            return C_f
    
    def kraus(self,n):
        return torch.tensor([[0,n[0],n[1],n[2]],
                        [-n[0],0,-n[2],n[1]],
                        [-n[1],n[2],0,-n[0]],
                        [-n[2],-n[1],n[0],0]],device=self.device,dtype=self.dtype_float)
    
    def complement(self,ix):
        self.ix_bool.fill_(False)
        self.ix_bool[ix]=True
        ix_bar = torch.nonzero(~self.ix_bool,as_tuple=True)[0]
        return ix_bar
    
    def generate_tripartite_circle(self,center=None,radius_factor=(2.6,2.6),shift=(0,0)):
        if center is None:
            center=[self.Lx/2,self.Ly/2]
        radius = [self.Lx/radius_factor[0],self.Ly/radius_factor[1]]
        
        A_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,Lx=self.Lx,Ly=self.Ly,center=center,radius=radius,angle=[0,np.pi/3*2]),shift=shift)
        B_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,Lx=self.Lx,Ly=self.Ly,center=center,radius=radius,angle=[np.pi/3*2,np.pi/3*4]),shift=shift)
        C_idx_0=self.linearize_idx_span(np.arange(self.Lx),np.arange(self.Ly),shape_func=lambda i,j: circle(i,j,Lx=self.Lx,Ly=self.Ly,center=center,radius=radius,angle=[np.pi/3*4,np.pi/3*6]),shift=shift)
        return torch.tensor(A_idx_0,device=self.device),torch.tensor(B_idx_0,device=self.device),torch.tensor(C_idx_0,device=self.device)
    
    
    def chern_number_quick(self,U1=True,shift=(0,0),selfaverage=False):
        # st=time.time()
        if selfaverage:
            return torch.stack([self.chern_number_quick(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            A_idx,B_idx,C_idx = self.generate_tripartite_circle(shift=shift)
            P=(torch.eye(self.C_m.shape[0],device=self.device,dtype=self.dtype_complex)-1j*self.C_m)/2
            P_AB=P[A_idx[:,None],B_idx[None,:]]
            P_BC=P[B_idx[:,None],C_idx[None,:]]
            P_CA=P[C_idx[:,None],A_idx[None,:]]
            P_AC=P[A_idx[:,None],C_idx[None,:]]
            P_CB=P[C_idx[:,None],B_idx[None,:]]
            P_BA=P[B_idx[:,None],A_idx[None,:]]
            h=-12*torch.pi*(torch.einsum("jk,kl,lj->jkl",P_AB,P_BC,P_CA)-torch.einsum("jl,lk,kj->jkl",P_AC,P_CB,P_BA)).imag
            # assert np.abs(h.imag).max()<1e-10, "Imaginary part of h is too large"
            nu=h.sum()
        # print('Chern number done in {:.4f}'.format(time.time()-st))
            if U1:
                return nu/2
            else:
                return nu

    def local_Chern_marker(self,Gamma,shift=[0,0],n_maj=2,U1=True):
        replica,layer,x,y,orbit,maj = np.unravel_index(np.arange(Gamma.shape[0]),(self.replica,self.layer,self.Lx,self.Ly,self.orbit,n_maj))
        x = torch.tensor((x+shift[0])%self.Lx,device=self.device)
        y = torch.tensor((y+shift[1])%self.Ly,device=self.device)
        C_f = self.get_C_f(Gamma,normal=False)
        xy_comm = torch.einsum("ij,j,jk,k,ki->i",C_f,x,C_f,y,C_f) - torch.einsum("ij,j,jk,k,ki->i",C_f,y,C_f,x,C_f)
        C_r = (xy_comm * 2 * torch.pi* 1j)
        C_r=C_r.reshape((self.replica,self.layer,self.Lx,self.Ly,self.orbit,n_maj))
        # assert np.abs(C_r.imag).max()<1e-10, f'imaginary part is {C_r.imag.max()}'
        if U1:
            return C_r.sum(axis=(-1,-2)).real/2
        else:
            return C_r.sum(axis=(-1,-2)).real

    def entanglement_contour(self,subregion,fermion=False, Gamma=None, fermion_idx=True,n=1):
        # c_A=self.c_subregion_m(subregion)
        c_A=self.c_subregion_m(subregion,Gamma,fermion_idx=fermion_idx)
        C_f=(torch.eye(c_A.shape[0],device=self.device)+1j*c_A)/2
        if n==1:
            f=self.xlogx(C_f,)
        if fermion:
            return torch.diag(f).real.reshape((-1,2)).sum(axis=1)
        else:
            return torch.diag(f).real
        
    def c_subregion_m(self,subregion,Gamma=None,fermion_idx=True):
        if Gamma is None:
            Gamma=self.C_m
        if fermion_idx:
            subregion=self.linearize_index(subregion,2)
        # return Gamma[np.ix_(subregion,subregion)]
        return Gamma[subregion[:,None],subregion[None,:]]
    def von_Neumann_entropy_m(self,subregion,Gamma=None,fermion_idx=True,verbose=False):
        st=time.time()
        c_A=self.c_subregion_m(subregion,Gamma,fermion_idx=fermion_idx)
        val=torch.linalg.eigvalsh(1j*c_A)
        val=(1-val)/2  
        val = val[(val>0) & (val<1)]
        if verbose:
            print('entanglement entropy done in {:.4f}'.format(time.time()-st))
        return -torch.sum(val*torch.log(val))-torch.sum((1-val)*torch.log(1-val))

    def half_cut_entanglement_entropy(self,shift=(0,0),selfaverage=False):
        if selfaverage:
            return torch.stack([self.half_cut_entanglement_entropy(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            Lx_first_half = (np.arange(self.Lx//2) + shift[0])%self.Lx
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            Lx_second_half = (np.arange(self.Lx//2,self.Lx) + shift[0])%self.Lx
            Ly_second_half = (np.arange(self.Ly//2,self.Ly) + shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_first_half,jlist=Ly_first_half)
            subB=self.c2g(ilist=Lx_first_half,jlist=Ly_second_half)
            subC=self.c2g(ilist=Lx_second_half,jlist=Ly_first_half)
            # subD=self.c2g(ilist=Lx_second_half,jlist=Ly_second_half)
            SAB=self.von_Neumann_entropy_m(torch.cat([subA,subB]),fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            return (SAB+SAC)/2

    def half_cut_entanglement_y_entropy(self,shift=(0,0),selfaverage=False):
        """half cut entanglement entropy with Lx x Ly/2"""
        if selfaverage:
            return torch.stack([self.half_cut_entanglement_entropy(shift=(0,j)) for j in range(self.Ly)]).mean()
        else:
            Lx_ = (np.arange(self.Lx))
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            Ly_second_half = (np.arange(self.Ly//2,self.Ly) + shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_,jlist=Ly_first_half)
            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            return SA



    def tripartite_mutual_information(self,shift=(0,0),selfaverage=False):
        """
        TMI uses four quadrants, covers both layer, [Assuming only one 1 replica]
        compute the tripartite mutual information as S(A)+S(B)+S(C)-S(AB)-S(BC)-S(AC)+S(ABC)
        """
        assert self.replica==1, "Tripartite mutual information only works for one replica"
        if selfaverage:
            return torch.stack([self.tripartite_mutual_information(shift=(i,j)) for i in range(self.Lx) for j in range(self.Ly)]).mean()
        else:
            Lx_first_half = (np.arange(self.Lx//2) + shift[0])%self.Lx
            Ly_first_half = (np.arange(self.Ly//2) +shift[1])%self.Ly
            Lx_second_half = (np.arange(self.Lx//2,self.Lx) + shift[0])%self.Lx
            Ly_second_half = (np.arange(self.Ly//2,self.Ly) + shift[1])%self.Ly
            subA=self.c2g(ilist=Lx_first_half,jlist=Ly_first_half)
            subB=self.c2g(ilist=Lx_first_half,jlist=Ly_second_half)
            subC=self.c2g(ilist=Lx_second_half,jlist=Ly_first_half)
            subD=self.c2g(ilist=Lx_second_half,jlist=Ly_second_half)

            SA=self.von_Neumann_entropy_m(subA,fermion_idx=False)
            SB=self.von_Neumann_entropy_m(subB,fermion_idx=False)
            SC=self.von_Neumann_entropy_m(subC,fermion_idx=False)
            SAB=self.von_Neumann_entropy_m(torch.cat([subA,subB]),fermion_idx=False)
            SBC=self.von_Neumann_entropy_m(torch.cat([subB,subC]),fermion_idx=False)
            SAC=self.von_Neumann_entropy_m(torch.cat([subA,subC]),fermion_idx=False)
            # SABC=self.von_Neumann_entropy_m(torch.cat([subA,subB,subC]),fermion_idx=False)
            SABC=self.von_Neumann_entropy_m(subD,fermion_idx=False)
            return SA+SB+SC-SAB-SBC-SAC+SABC
    
    def c2g(self,ilist,jlist):
        # ilist=np.arange(0,self.Lx//2)
        # jlist=np.arange(0,self.Ly)
        return torch.hstack((
            torch.from_numpy(self.linearize_idx_span(ilist = ilist,jlist=jlist,layer=0)).cuda(),
            torch.from_numpy(self.linearize_idx_span(ilist = ilist,jlist=jlist,layer=1)).cuda())
        )

    def xlogx(self,A):
        val,vec=torch.linalg.eigh(A)
        negative=val<=0
        val[negative]=0
        val_pos=val[~negative]
        val[~negative]=-val_pos*torch.log(val_pos)
        val=val+0j
        return vec@torch.diag(val)@vec.conj().T

    def C_m_selfaverage(self,n=1):
        """take the selfaverage of Gamma, by shifting all possible coordinates"""
        idx=(self.replica, self.layer, self.Lx*self.Ly,self.orbit,2)
        C_m_reshape=(self.C_m**n).view(idx*2)
        d_list =[1]*len(idx*2)
        Lxy=self.Lx*self.Ly
        d_list[2]=d_list[7]=Lxy
        kernel = torch.eye(Lxy,device=self.device).reshape(d_list)/(Lxy)
        return torch.fft.ifft2(torch.fft.fft2(C_m_reshape,dim=(2,7))*torch.fft.fft2(kernel,dim=(2,7)),dim=(2,7)).reshape(self.C_m.shape).real




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

