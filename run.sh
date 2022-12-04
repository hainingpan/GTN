mpirun -n 8 python -m mpi4py.futures run.py --Born --es 10 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 3 --b2 1 1 1 --L 64 
mpirun -n 4 python -m mpi4py.futures run.py --Born --es 10 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 3 --b2 1 1 1 --L 64 
mpirun -n 2 python -m mpi4py.futures run.py --Born --es 10 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 3 --b2 1 1 1 --L 64 

# python run.py --Born --es 2 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .9 3 --b2 1 1 1 --L 64 
