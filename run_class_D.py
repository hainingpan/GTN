from GTN import *
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from mpi4py.futures import MPIPoolExecutor

def run_single_class_D(inputs):
    # print(inputs)
    L,seed,p,t,r=inputs
    gtn=GTN(L=L,seed=seed,history=False,op=False,random_init=False)
    for i in range(L):
        gtn.measure_all_tri_op_D(p_list=p,r_list=r,even=True,sigma=1/8)
        gtn.measure_all_tri_op_D(p_list=1-p,r_list=r,even=False,sigma=1/8)
    MI=gtn.mutual_information_cross_ratio(unitcell=1,ratio=[1,4])
    EE=gtn.von_Neumann_entropy_m_self_average(unitcell=1)
    return MI,EE

def wrapper(inputs):
    try:
        MI,EE=run_single_class_D(inputs)
        return MI,EE
    except:
        return np.nan,np.nan

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=6,type=int)
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--t','-t',type=int,default=None)
    parser.add_argument('--p','-p',type=float,nargs=3,default=[.5,.5,1],help="probablity for measuring even pairs")
    parser.add_argument('--r','-r',type=int,default=None)

    args=parser.parse_args()
    t=args.t if args.t is not None else args.L
    p_list=np.linspace(*args.p[:2],int(args.p[2]))

    st=time.time()
    inputs=[(args.L,seed,p,t,args.r) for p in p_list for seed in range(args.es)]
    with MPIPoolExecutor() as executor:
        rs=list(tqdm(executor.map(wrapper,inputs),total=len(inputs)))
    # rs=list(tqdm(map(wrapper,inputs),total=len(inputs)))
    rs=np.array(rs).reshape((int(args.p[2]),args.es,2))
    MI,EE=rs[:,:,0],rs[:,:,1]
    
    with open('class_D_p({:.2f},{:.2f},{:.0f})_En{:d}_L{:d}_t{:d}_r{:d}.pickle'.format(*args.p,args.es,args.L,t,args.r),'wb') as f:
        pickle.dump({"args":args,"MI":MI,"EE":EE}, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))


