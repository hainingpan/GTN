# python run_classA_2D.py --L 15 --nshell 1 --mu 1
# python run_classA_2D.py --L 15 --nshell 2 --mu 1
# python run_classA_2D.py --L 15 --nshell 3 --mu 1
# python run_classA_2D.py --L 15 --nshell 4 --mu 1

# python run_classA_2D.py --L 21 --nshell 2 --mu 0.4
# python run_classA_2D.py --L 21 --nshell 2 --mu 0.8
# python run_classA_2D.py --L 21 --nshell 2 --mu 1.2
# python run_classA_2D.py --L 21 --nshell 2 --mu 1.6
# python run_classA_2D.py --L 21 --nshell 2 --mu 0.2
# python run_classA_2D.py --L 21 --nshell 2 --mu 0.6
# python run_classA_2D.py --L 21 --nshell 2 --mu 1.4
# python run_classA_2D.py --L 21 --nshell 2 --mu 1.8



# python run_classA_2D.py --L 18 --nshell 1 --mu 1
# python run_classA_2D.py --L 18 --nshell 2 --mu 1
# python run_classA_2D.py --L 18 --nshell 3 --mu 1
# python run_classA_2D.py --L 18 --nshell 4 --mu 1

# python run_classA_2D.py --L 21 --nshell 1 --mu 1
# python run_classA_2D.py --L 21 --nshell 2 --mu 1
# python run_classA_2D.py --L 21 --nshell 3 --mu 1
# python run_classA_2D.py --L 21 --nshell 4 --mu 1

# mpirun -np 3 python -m mpi4py.futures run_classA_2D.py --L 10 --nshell 2 --mu 1 --es 2
# mpirun -np 11 python -m mpi4py.futures run_classA_2D.py --L 10 --nshell 2 --mu 1 --es 10
# mpirun -np 11 python -m mpi4py.futures run_classA_2D.py --L 12 --nshell 2 --mu 1 --es 10
# mpirun -np 11 python -m mpi4py.futures run_classA_2D.py --L 14 --nshell 2 --mu 1 --es 10
# mpirun -np 11 python -m mpi4py.futures run_classA_2D.py --L 16 --nshell 2 --mu 1 --es 10
# mpirun -np 11 python -m mpi4py.futures run_classA_2D.py --L 18 --nshell 2 --mu 1 --es 10
# mpirun -np 11 python -m mpi4py.futures run_classA_2D.py --L 20 --nshell 2 --mu 1 --es 10


python run_classA_2D_DW.py --L 10 --nshell 2 --tf 10  --truncate