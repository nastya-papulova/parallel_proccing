import re
import time

import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import cg


def read_data_info(filename):
    """
    Reads the size of matrix A and the length of vector x from a data file.

    Parameters:
    filename (str): The path to the data file.

    Returns:
    tuple: A tuple of two integers containing the size of matrix A and the integer length of vector x.

    Raises:
    ValueError: If vector and matrix cannot be multiplied due to incorrect dimensions.
    ValueError: If the expected data format is not found in the file.
    """
    with open(filename, "r") as file:
        content = file.read()

    matrix_A_size_match = re.search(r"matrix_A_size\s*=\s*\((\d+),\s*(\d+)\)", content)
    vector_x_len_match = re.search(r"vector_x_len\s*=\s*(\d+)", content)

    if matrix_A_size_match and vector_x_len_match:
        matrix_A_size = (
            int(matrix_A_size_match.group(1)),
            int(matrix_A_size_match.group(2)),
        )
        vector_x_len = int(vector_x_len_match.group(1))
        if vector_x_len != matrix_A_size[1]:
            raise ValueError(
                "Vector and matrix cannot be multiplied due to incorrect dimensions."
            )

        return matrix_A_size, vector_x_len


def sequential_calculation_conjugate_gradient_method(A, b, x, N):
    s = 1
    p = np.zeros(N)

    while s <= N:
        if s == 1:
            r = np.dot(A.T, np.dot(A, x) - b)
        else:
            r -= q / np.dot(p, q)

        p += r / np.dot(r, r)
        q = np.dot(A.T, np.dot(A, p))
        x -= p / np.dot(p, q)
        s += 1
    return x


def sequential_conjugate_gradient_method(
    matrix_A_size,
    matrix_A_data_filename,
    vector_b_data_filename,
):
    """
    Performs regular (sequential) conjugate gradient method for solving SLAEs.

    Parameters:
    matrix_A_size (tuple): The size of matrix A.
    vector_x_len (int): The length of vector x.
    matrix_A_data_filename (str): The filename containing the data for matrix A.
    vector_x_data_filename (str): The filename containing the data for vector x.

    Returns:
    np.ndarray: The resulting vector of SLAE solution.
    """
    M, N = matrix_A_size
    matrix_A = np.loadtxt(matrix_A_data_filename).reshape(matrix_A_size)
    vector_b = np.loadtxt(vector_b_data_filename)

    x = np.zeros(N)
    x = sequential_calculation_conjugate_gradient_method(matrix_A, vector_b, x, N)

    return x


def auxiliary_arrays_determination(M, numprocs):
    displs = np.empty(numprocs, dtype=np.int32)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs[0] = 0
    rcounts[0] = 0
    ave, res = divmod(M, numprocs - 1)
    rcounts[1:] = [int(ave + 1 if i <= res else ave) for i in range(1, numprocs)]
    displs[1:] = np.cumsum(rcounts[:-1])

    return rcounts, displs


def parallel_calculation_conjugate_gradient_method(A_part, b_part, x, N):
    r = np.empty(N, dtype=np.float64)
    q = np.empty(N, dtype=np.float64)

    s = 1
    p = np.zeros(N, dtype=np.float64)

    while s <= N:
        if s == 1:
            r_temp = np.dot(A_part.T, np.dot(A_part, x) - b_part)
            comm.Allreduce([r_temp, N, MPI.DOUBLE], [r, N, MPI.DOUBLE], op=MPI.SUM)
        else:
            r -= q / np.dot(p, q)
        p += r / np.dot(r, r)
        q_temp = np.dot(A_part.T, np.dot(A_part, p))
        comm.Allreduce([q_temp, N, MPI.DOUBLE], [q, N, MPI.DOUBLE], op=MPI.SUM)

        x -= p / np.dot(p, q)
        s += 1
    return x


def parallel_conjugate_gradient_method(
    comm,
    numprocs,
    rank,
    matrix_A_size,
    vector_b_len,
    matrix_A_data_filename,
    vector_b_data_filename,
):
    """
    Performs parallel conjugate gradient method for solving SLAEs.

    Parameters:
    comm (MPI.Comm): The MPI communicator.
    numprocs (int): The number of processes.
    rank (int): The rank of the current process.
    matrix_A_size (tuple): The size of matrix A.
    vector_b_len (int): The length of vector b.
    matrix_A_data_filename (str): The filename containing the data for matrix A.
    vector_b_data_filename (str): The filename containing the data for vector b.

    Returns:
    np.ndarray: The resulting vector of SLAE solution.
    """

    if rank == 0:
        M, N = matrix_A_size
        M, N = np.array(M), np.array(N)
    else:
        N = np.array(0, dtype=np.int32)

    # Broadcast the number of columns in matrix A to all processes
    comm.Bcast([N, 1, MPI.INT], root=0)

    if rank == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    else:
        rcounts_M, displs_M = None, None

    M_part = np.array(0, dtype=np.int32)
    comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

    if rank == 0:
        with open(matrix_A_data_filename, "r") as file:
            matrix_A = file.readlines()

        matrix_A = np.array(matrix_A, dtype=np.float64).reshape(M, N)

        for k in range(1, numprocs):
            A_part = matrix_A[displs_M[k] : displs_M[k] + rcounts_M[k], :]
            comm.Send([A_part, MPI.DOUBLE], dest=k, tag=0)

        A_part = matrix_A[displs_M[rank] : displs_M[rank] + rcounts_M[rank], :]
    else:
        A_part = np.empty((M_part, N), dtype=np.float64)
        comm.Recv([A_part, MPI.DOUBLE], source=0, tag=0)

    # Allocate space for the vector b
    vector_b = np.empty(vector_b_len, dtype=np.float64)

    if rank == 0:
        vector_b = np.loadtxt(vector_b_data_filename)
    else:
        vector_b = None

    b_part = np.empty(M_part, dtype=np.float64)
    comm.Scatterv(
        [vector_b, rcounts_M, MPI.DOUBLE], [b_part, M_part, MPI.DOUBLE], root=0
    )

    x = np.zeros(N, dtype=np.float64)

    x = parallel_calculation_conjugate_gradient_method(A_part, b_part, x, N)

    return x


data_info_filename = "Data/data_info.dat"
vector_b_data_filename = "Data/vector_x.dat"
matrix_A_data_filename = "Data/matrix_A.dat"

matrix_A_size, vector_b_len = read_data_info(data_info_filename)

# Initialize MPI
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

start_time = time.time()

x_parallel = parallel_conjugate_gradient_method(
    comm,
    numprocs,
    rank,
    matrix_A_size,
    vector_b_len,
    matrix_A_data_filename,
    vector_b_data_filename,
)
end_time = time.time()

if rank == 0:
    print(f"Parallel conjugate gradient method time: {end_time - start_time} seconds")


start_time = time.time()

x_sequential = sequential_conjugate_gradient_method(
    matrix_A_size,
    matrix_A_data_filename,
    vector_b_data_filename,
)
end_time = time.time()

if rank == 0:
    print(f"Sequential conjugate gradient method time: {end_time - start_time} seconds")
    print(np.allclose(x_parallel, x_sequential))
