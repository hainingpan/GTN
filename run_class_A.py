from GTN import *
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor


def run_single_class_AIII(inputs):
    L,seed,vartheta,t,Born=inputs
    A=np.cos(vartheta)
    gtn=GTN(L=L,seed=seed,history=False,op=False,random_init=False)
    sites_flip=np.sort(gtn.rng.choice(np.arange(gtn.L),size=gtn.L//2,replace=False))
    gtn.measure([[1,0,0]]*(gtn.L//2),np.sort(np.r_[2*sites_flip,2*sites_flip+1]))
    for i in range(t):
        gtn.measure_all_class_AIII(A_list=A,Born=Born,class_A=True,even=True,)
        gtn.measure_all_class_AIII(A_list=np.sqrt(1-A**2),Born=Born,class_A=True,even=False,)
    MI=gtn.mutual_information_cross_ratio()
    return MI

def wrapper(inputs):
    # a1,a2,b1,b2,es,L,t,Born=inputs
    MI=run_single_class_AIII(inputs)
    return MI

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=6,type=int)
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--t','-t',type=int,default=None)
    parser.add_argument('--vartheta','-vartheta',type=float,nargs=3,default=[.5,.5,1],help="the angle will be multipled with pi, i.e. vartheta=vartheta*np.pi")
    parser.add_argument('-Born','--Born',action='store_true',help='set for Born rule')
    args=parser.parse_args()
    t=args.t if args.t is not None else args.L
    vartheta_list=np.linspace(*args.vartheta[:2],int(args.vartheta[2]))*np.pi

    st=time.time()
    inputs=[(args.L,seed,vartheta,t,args.Born) for vartheta in vartheta_list for seed in range(args.es)]
    with MPIPoolExecutor() as executor:
        rs=list(tqdm(executor.map(wrapper,inputs),total=len(inputs)))
    # rs=list(tqdm(map(wrapper,inputs),total=len(inputs)))
    MI=rs
    MI=np.array(MI).reshape((int(args.vartheta[2]),args.es))
    with open('class_A_vartheta({:.2f},{:.2f},{:.0f})_En{:d}_L{:d}_t{:d}.pickle'.format(*args.vartheta,args.es,args.L,t),'wb') as f:
        pickle.dump({"args":args,"MI":MI}, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))


