from mpi4py import MPI
from numpy import empty

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

print("hello from process {0} out of {1}".format(rank, numprocs))