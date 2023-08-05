from GTN import *
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor

def measure(inputs):
    p,es,L,t,Born=inputs
    gtn=GTN(L=L,seed=es,history=False,random_init=False)
    MI=[]
    for i in range(t):
        gtn.measure_all_tri_op(p_list=p,Born=Born,even=True)
        gtn.measure_all_tri_op(p_list=1-p,Born=Born,even=False)
        if i>=gtn.L-1:
            MI.append(gtn.mutual_information_cross_ratio())
    return np.array(MI)

def wrapper(inputs):
    p,es,L,t,Born=inputs
    MI=measure(inputs)
    return MI

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=6,type=int)
    parser.add_argument('--L','-L',type=int,default=16)
    parser.add_argument('--t','-t',type=int,default=None)
    parser.add_argument('--p','-p',type=float,nargs=3,default=[0,1,11])
    parser.add_argument('-Born','--Born',action='store_true',help='set for Born rule')
    args=parser.parse_args()
    t=args.t if args.t is not None else args.L

    st=time.time()
    inputs=[(p,es,args.L,t,args.Born) for p in np.linspace(*args.p[:2],int(args.p[2])) for es in range(args.es)]
    with MPIPoolExecutor() as executor:
        rs=list(executor.map(wrapper,inputs))
    # rs=list(map(wrapper,inputs))

    MI=rs
    MI=np.array(MI).reshape((int(args.p[2]),args.es,t-args.L+1))
    with open('GTN_p({:.2f},{:.2f},{:.0f})_En{:d}_L{:d}_t{:d}_{:s}.pickle'.format(*args.p,args.es,args.L,t,'Born' if args.Born else 'Forced'),'wb') as f:
        pickle.dump({"args":args,"MI":MI}, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))


