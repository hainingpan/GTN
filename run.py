from GTN import *
import time
import argparse
import pickle
import numpy as np
from mpi4py.futures import MPIPoolExecutor

def measure(inputs):
    a1,a2,b1,b2,es,L,t,Born=inputs
    gtn=GTN(L=L,seed=es,history=False)
    for i in range(t):
        gtn.measure_all_sync(b2=b2,a2=a2, a1=a1,b1=b1,Born=Born,even=True,n1_z=True)
        gtn.measure_all_sync(b2=b2,a2=a2, a1=a1,b1=b1,Born=Born,even=False,n1_z=False)
    return gtn.mutual_information_cross_ratio()

def wrapper(inputs):
    a1,a2,b1,b2,es,L,t,Born=inputs
    MI=measure(inputs)
    return MI

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--es','-es',default=6,type=int)
    parser.add_argument('--L','-L',type=int)
    parser.add_argument('--t','-t',type=int,default=None)
    parser.add_argument('--a1','-a1',type=float,nargs=3,default=[.5,.5,1])
    parser.add_argument('--b1','-b1',type=float,nargs=3,default=[1,1,1])
    parser.add_argument('--a2','-a2',type=float,nargs=3,default=[0,1,11])
    parser.add_argument('--b2','-b2',type=float,nargs=3,default=[1,1,1])
    parser.add_argument('-Born','--Born',action='store_true',help='set for Born rule')
    args=parser.parse_args()
    t=args.t if args.t is not None else args.L

    st=time.time()
    inputs=[(a1,a2,b1,b2,es,args.L,t,args.Born) for a1 in np.linspace(*args.a1[:2],int(args.a1[2])) for a2 in np.linspace(*args.a2[:2],int(args.a2[2])) for b1 in np.linspace(*args.b1[:2],int(args.b1[2])) for b2 in np.linspace(*args.b2[:2],int(args.b2[2])) for es in range(args.es)]
    with MPIPoolExecutor() as executor:
        rs=list(executor.map(wrapper,inputs))
    # rs=list(map(wrapper,inputs))

    MI=rs
    MI=np.array(MI).reshape((int(args.a1[2]),int(args.a2[2]),int(args.b1[2]),int(args.b2[2]),args.es))
    with open('GTN_a1({:.2f},{:.2f},{:.0f})_a2({:.2f},{:.2f},{:.0f})_b1({:.2f},{:.2f},{:.0f})_b2({:.2f},{:.2f},{:.0f})_En{:d}_L{:d}_t{:d}.pickle'.format(*args.a1,*args.a2,*args.b1,*args.b2,args.es,args.L,t),'wb') as f:
        pickle.dump({"args":args,"MI":MI}, f)

    print('Time elapsed: {:.4f}'.format(time.time()-st))


