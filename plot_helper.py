import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex']=False
plt.rcParams['font.family']='serif'
plt.rcParams['font.size']=9
plt.rcParams['axes.titlesize']=plt.rcParams['font.size']
plt.rcParams['figure.figsize']=(6.8,4)
# plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath,amssymb,physics,bm}'

def plot_chern(nu_list,log=False,C=0,ax=None,label=None,color=None):
    """plot chern number"""
    if ax is None:
        fig,ax=plt.subplots()
    if log:
        ax.plot(1+np.arange(len(nu_list)),np.abs(np.abs(np.array(nu_list))-C),'.-',label=label,color=color)
    else:
        ax.plot(np.arange(len(nu_list)),(np.array(nu_list)),'.-',label=label,color=color)

    ax.set_xlabel('epoch')
    if log:
        ax.set_ylabel(rf'$||{{\mathcal{{C}}}}|-{C}|$')
        ax.set_yscale('log')
        ax.set_xscale('log')
    else:
        ax.set_ylabel(r'${\mathcal{C}}$')

def plot_C_r(C_r,ax=None,bottomcb=True,label_pos=None):
    """plot Chern marker"""
    if ax is None:
        fig,ax=plt.subplots(1,2,figsize=(5,2.5))
    im0=ax[0].imshow(np.tanh(C_r[0]),cmap='bwr',vmin=-1,vmax=1)
    place_color_bar_top(im0,ax[0],cbticks=[-1,0,1],cblabels=[-1,0,1],label=r'$\tanh\mathcal{C}_\mathfrak{t}(\mathbf{r})$')
    im1=ax[1].imshow(np.tanh(C_r[1]),cmap='bwr',vmin=-1,vmax=1)
    if bottomcb:
        print(bottomcb)
        place_color_bar_top(im1,ax[1],cbticks=[-1,0,1],cblabels=[-1,0,1],label=r'$\tanh\mathcal{C}_\mathfrak{b}(\mathbf{r})$')
        
    else:
        ax[1].set_title(r'$\tanh\mathcal{C}_\mathfrak{b}(\mathbf{r})$')
        pass
    if label_pos is not None:
        if label_pos[0] is not None:
            x,y=label_pos[0]
            cross0= ax[0].scatter(y,x,s=50,marker='x',color='k')
        else:
            cross0= ax[0].scatter(0,0,s=0,marker='x',color='k')

        if label_pos[1] is not None:
            x,y=label_pos[1]
            cross1= ax[1].scatter(y,x,s=50,marker='x',color='k')
        else:
            cross1= ax[1].scatter(0,0,s=0,marker='x',color='k')
    else:
        cross0= ax[0].scatter(0,0,s=0,marker='x',color='k')
        cross1= ax[1].scatter(0,0,s=0,marker='x',color='k')


    return im0,im1,cross0,cross1
def place_color_bar_top(im,ax,label,cbticks=[-np.pi,0,np.pi],cblabels=[r'$-\pi$',r'$0$',r'$\pi$']):
    axins=ax.inset_axes([0.,1.1,1,0.1])
    cb=plt.colorbar(im,cax=axins,orientation='horizontal')
    if cblabels is not None: 
        cb.set_label(label,)
    if cbticks is not None:
        cb.set_ticks(cbticks,labels=cblabels)
    cb.ax.xaxis.set_ticks_position('top')
    return cb

def plot_EC(EC,ax=None,bottomcb=True,label_pos=None,vmax=2):
    if ax is None:
        fig,ax=plt.subplots(1,2,figsize=(5,2.5))
    im0=ax[0].imshow((EC[0])/np.log(2),cmap='Blues',vmin=0,vmax=vmax)
    cb0,cb1=None,None
    if vmax is None:
        cb0=place_color_bar_top(im0,ax[0],cbticks=None,cblabels=None,label=r'$s(\mathbf{r})$')
    else:
        cb0=place_color_bar_top(im0,ax[0],cbticks=np.linspace(0,vmax,3),cblabels=np.linspace(0,vmax,3),label=r'$s(\mathbf{r})$')
    im1=ax[1].imshow((EC[1])/np.log(2),cmap='Blues',vmin=0,vmax=vmax)
    if bottomcb:
        if vmax is None:
            cb1=place_color_bar_top(im1,ax[1],cbticks=None,cblabels=None,label=r'$s_\mathfrak{b}(\mathbf{r})$')
        else:
            cb1=place_color_bar_top(im1,ax[1],cbticks=np.linspace(0,vmax,3),cblabels=np.linspace(0,vmax,3),label=r'$s_\mathfrak{b}(\mathbf{r})$')
    else:
        ax[1].set_title(r'$s_\mathfrak{b}(\mathbf{r})$')
    if label_pos is not None:
        if label_pos[0] is not None:
            x,y=label_pos[0]
            cross0= ax[0].scatter(y,x,s=50,marker='x',color='k')
        else:
            cross0= ax[0].scatter(0,0,s=0,marker='x',color='k')

        if label_pos[1] is not None:
            x,y=label_pos[1]
            cross1= ax[1].scatter(y,x,s=50,marker='x',color='k')
        else:
            cross1= ax[1].scatter(0,0,s=0,marker='x',color='k')
    else:
        cross0= ax[0].scatter(0,0,s=0,marker='x',color='k')
        cross1= ax[1].scatter(0,0,s=0,marker='x',color='k')


    return im0,im1,cross0,cross1, cb0,cb1

def convert_to_list(a_i):
    i_list=[]
    j_list=[]
    a_list=[]
    for key,value in a_i.items():
        i_list.append(key[0])
        j_list.append(key[1])
        a_list.append(value)
    a_list=np.array(a_list)
    i_list=np.array(i_list)
    j_list=np.array(j_list)
    return i_list,j_list,a_list

def plot_eigvals(C_m_,ax=None):
    import torch
    if ax is None:
        fig,ax=plt.subplots(figsize=(4,3))
    # C_m_=gtn2_torch_0_list[0].C_m_selfaverage()
    L = C_m_.shape[0]//2
    C_m_t=C_m_[:L,:L]
    C_m_b=C_m_[L:,L:]

    eigval_t=torch.linalg.eigvalsh(C_m_t/1j)
    eigval_b=torch.linalg.eigvalsh(C_m_b/1j)
    eigvals=torch.concat([eigval_t,eigval_b])
    color_list = np.array(['b']*eigval_t.shape[0]+['r']*eigval_b.shape[0])
    order=eigvals.argsort()
    color_list=color_list[order.cpu()]
    eigvals_s=eigvals[order].cpu()
    ax.plot(torch.linalg.eigvalsh(C_m_/1j).cpu(),'.',color='gray')
    ax.scatter(np.arange((eigvals_s).shape[0]),eigvals_s.cpu(),color=color_list)
    ax.set_xlabel('index')
    ax.set_ylabel('Eigenvalues')
    ax.set_title(f'Spectral Gap: Top {torch.abs(eigval_t).min():.3f} Bottom {torch.abs(eigval_b).min():.3f}')