import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex']=False
# plt.rcParams['font.family']='serif'
# plt.rcParams['font.size']=9
# plt.rcParams['axes.titlesize']=plt.rcParams['font.size']
# plt.rcParams['figure.figsize']=(6.8,4)
# plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath,amssymb,physics,bm}'

def plot_chern(nu_list,log=False,C=0,ax=None,label=None,color=None):
    """plot chern number"""
    if ax is None:
        fig,ax=plt.subplots()
    if log:
        ax.plot(1+np.arange(len(nu_list)),np.abs(np.abs(np.array(nu_list))-C),'.-',label=label,color=color)
    else:
        ax.plot(np.arange(len(nu_list)),(np.array(nu_list)),'.-',label=label,color=color)

    # ax.set_xlabel('steps')
    ax.set_xlabel('epoch')
    if log:
        ax.set_ylabel(rf'$\abs{{\abs{{\mathcal{{C}}}}-{C}}}$')
        ax.set_yscale('log')
        ax.set_xscale('log')
    else:
        ax.set_ylabel(r'${\mathcal{C}}$')

def plot_C_r(C_r,ax=None,bottomcb=True,label_pos=None):
    """plot Chern marker"""
    if ax is None:
        fig,ax=plt.subplots(1,2,figsize=(5,2.5))
    im0=ax[0].imshow(np.tanh(C_r[0]),cmap='bwr',vmin=-1,vmax=1)
    place_color_bar_top(im0,ax[0],cbticks=[-1,0,1],cblabels=[-1,0,1],label=r'$\tanh\mathcal{C}_\mathfrak{b}(r)$')
    # label=r'$\tanh\mathcal{C}_\mathfrak{t}(\bm{r})$'
    im1=ax[1].imshow(np.tanh(C_r[1]),cmap='bwr',vmin=-1,vmax=1)
    if bottomcb:
        print(bottomcb)
        place_color_bar_top(im1,ax[1],cbticks=[-1,0,1],cblabels=[-1,0,1],label=r'$\tanh\mathcal{C}_\mathfrak{b}(r)$')
        # label=r'$\tanh\mathcal{C}_\mathfrak{b}(\bm{r})$'
    else:
        # ax[1].set_title(r'$\tanh\mathcal{C}_\mathfrak{b}(\bm{r})$')
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
        cb0=place_color_bar_top(im0,ax[0],cbticks=None,cblabels=None,label=r'$s(\bm{r})$')
    else:
        cb0=place_color_bar_top(im0,ax[0],cbticks=np.linspace(0,vmax,3),cblabels=np.linspace(0,vmax,3),label=r'$s(\bm{r})$')
    im1=ax[1].imshow((EC[1])/np.log(2),cmap='Blues',vmin=0,vmax=vmax)
    if bottomcb:
        if vmax is None:
            cb1=place_color_bar_top(im1,ax[1],cbticks=None,cblabels=None,label=r'$s_\mathfrak{b}(\bm{r})$')
        else:
            cb1=place_color_bar_top(im1,ax[1],cbticks=np.linspace(0,vmax,3),cblabels=np.linspace(0,vmax,3),label=r'$s_\mathfrak{b}(\bm{r})$')
    else:
        ax[1].set_title(r'$s_\mathfrak{b}(\bm{r})$')
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
