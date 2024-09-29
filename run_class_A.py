from GTN import *
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor


def run_single_class_AIII(inputs):
    L,seed,vartheta,t,Born,class_A=inputs
    A=np.cos(vartheta)
    gtn=GTN(L=L,seed=seed,history=False,op=False,random_init=False)
    sites_flip=np.sort(gtn.rng.choice(np.arange(gtn.L),size=gtn.L//2,replace=False))
    gtn.C_m[sites_flip*2,sites_flip*2+1]=-1
    gtn.C_m[sites_flip*2+1,sites_flip*2]=1
    for i in range(t):
        gtn.measure_all_class_AIII(A_list=A,Born=Born,class_A=class_A,even=True,)
        gtn.measure_all_class_AIII(A_list=np.sqrt(1-A**2),Born=Born,class_A=class_A,even=False,)
    MI=gtn.mutual_information_cross_ratio(unitcell=2)
    EE=gtn.von_Neumann_entropy_m_self_average(unitcell=2)
    return MI,EE

def run_single_class_AIII_unitary(inputs):
    L,seed,vartheta,t,r,Born,class_A=inputs
    A=np.cos(vartheta)
    gtn=GTN(L=L,seed=seed,history=False,op=False,random_init=False)
    sites_flip=np.sort(gtn.rng.choice(np.arange(gtn.L),size=gtn.L//2,replace=False))
    gtn.C_m[sites_flip*2,sites_flip*2+1]=-1
    gtn.C_m[sites_flip*2+1,sites_flip*2]=1
    for i in range(t):
        # gtn.measure_all_class_AIII_r_unified(A_list=A,Born=Born,r_list=r,even=True,class_A=True)
        # gtn.measure_all_class_AIII_r_unified(A_list=np.sqrt(1-A**2),Born=Born,r_list=r,even=False,class_A=True)
        gtn.measure_all_class_AIII_r_unitary(A_list=A,Born=Born,r_list=r,even=True,class_A=True)
        gtn.measure_all_class_AIII_r_unitary(A_list=np.sqrt(1-A**2),r_list=r,Born=Born,even=False,class_A=True)
    MI=gtn.mutual_information_cross_ratio(unitcell=2,ratio=[1,8])
    EE=gtn.von_Neumann_entropy_m_self_average(unitcell=2)
    return MI,EE

def wrapper(inputs):
    # a1,a2,b1,b2,es,L,t,Born=inputs
    try:
        # MI,EE=run_single_class_AIII(inputs)
        MI,EE=run_single_class_AIII_unitary(inputs)
        return MI,EE
    except:
        return np.nan,np.nan

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=6,type=int)
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--t','-t',type=int,default=None)
    parser.add_argument('--vartheta','-vartheta',type=float,nargs=3,default=[.5,.5,1],help="the angle will be multipled with pi, i.e. vartheta=vartheta*np.pi")
    parser.add_argument('-Born','--Born',action='store_true',help='set for Born rule')
    parser.add_argument('-class_A','--class_A',action='store_true',help='set for class AIII rule')
    parser.add_argument('--r','-r',type=int,default=None)

    args=parser.parse_args()
    t=args.t if args.t is not None else args.L
    vartheta_list=np.linspace(*args.vartheta[:2],int(args.vartheta[2]))*np.pi

    st=time.time()
    # inputs=[(args.L,seed,vartheta,t,args.Born,args.class_A) for vartheta in vartheta_list for seed in range(args.es)]
    inputs=[(args.L,seed,vartheta,t,args.r,args.Born,args.class_A) for vartheta in vartheta_list for seed in range(args.es)]
    with MPIPoolExecutor() as executor:
        rs=list(tqdm(executor.map(wrapper,inputs),total=len(inputs)))
    # rs=list(tqdm(map(wrapper,inputs),total=len(inputs)))
    rs=np.array(rs).reshape((int(args.vartheta[2]),args.es,2))
    MI,EE=rs[:,:,0],rs[:,:,1]
    
    # with open('class_{}_vartheta({:.2f},{:.2f},{:.0f})_En{:d}_L{:d}_t{:d}.pickle'.format('A' if args.class_A else 'AIII',*args.vartheta,args.es,args.L,t),'wb') as f:
    with open('class_{}_vartheta({:.2f},{:.2f},{:.0f})_En{:d}_L{:d}_t{:d}_r{:d}.pickle'.format('A' if args.class_A else 'AIII',*args.vartheta,args.es,args.L,t,args.r),'wb') as f:
        pickle.dump({"args":args,"MI":MI,"EE":EE}, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))


