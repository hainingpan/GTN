# mpirun -n 32 python -m mpi4py.futures run.py --Born --es 50 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 21 --b2 1 1 1 --L 64 
# mpirun -n 32 python -m mpi4py.futures run.py --Born --es 50 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 21 --b2 1 1 1 --L 128 
# mpirun -n 32 python -m mpi4py.futures run.py --Born --es 50 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 21 --b2 1 1 1 --L 256 
# mpirun -n 16 python -m mpi4py.futures run.py --Born --es 50 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 21 --b2 1 1 1 --L 64 
# mpirun -n 8 python -m mpi4py.futures run.py --Born --es 50 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 21 --b2 1 1 1 --L 64 

# python run.py --Born --es 2 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .9 3 --b2 1 1 1 --L 128 
# python run.py --Born --es 50 --a1 .5 .5 1 --b1 1 1 1 --a2 0 .99 21 --b2 1 1 1 --L 128 

mpirun -n 20 python -m mpi4py.futures run_class_A_r.py --Born --es 1000 --vartheta 0.49 0.49 1 --L 8 --r 0 1 11
mpirun -n 20 python -m mpi4py.futures run_class_A_r.py --Born --es 1000 --vartheta 0.49 0.49 1 --L 16 --r 0 1 11
mpirun -n 20 python -m mpi4py.futures run_class_A_r.py --Born --es 1000 --vartheta 0.49 0.49 1 --L 32 --r 0 1 11
mpirun -n 20 python -m mpi4py.futures run_class_A_r.py --Born --es 1000 --vartheta 0.49 0.49 1 --L 64 --r 0 1 11

