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

    matrix_A = np.loadtxt(matrix_A_data_filename).reshape(matrix_A_size)
    vector_b = np.loadtxt(vector_b_data_filename)

    x = cg(matrix_A, vector_b)

    return x


def auxiliary_arrays_determination(M, numprocs):
    displs = np.empty(numprocs, dtype=np.int32)
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs[0] = 0
    rcounts[0] = 0
    ave, res = divmod(M, numprocs - 1)
    rcounts[1:] = [int(ave + 1 if i <= res else ave) for i in range(1, numprocs)]
    displs[1:] = np.cumsum(rcounts[:-1])


def parallel_conjugate_gradient_method(
    comm,
    numprocs,
    rank,
    matrix_A_size,
    vector_x_len,
    matrix_A_data_filename,
    vector_x_data_filename,
):
    """
    Performs parallel transposed matrix-vector multiplication using MPI.

    Parameters:
    comm (MPI.Comm): The MPI communicator.
    numprocs (int): The number of processes.
    rank (int): The rank of the current process.
    matrix_A_size (tuple): The size of matrix A.
    vector_x_len (int): The length of vector x.
    matrix_A_data_filename (str): The filename containing the data for matrix A.
    vector_x_data_filename (str): The filename containing the data for vector x.

    Returns:
    np.ndarray: The resulting vector after multiplication (only for the root process).
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
        rcounts_N, displs_N = auxiliary_arrays_determination(N, numprocs)
    else:
        rcounts_M, displs_M = None, None
        rcounts_N = np.empty(numprocs, dtype=np.int32)
        displs_N = np.empty(numprocs, dtype=np.int32)

    comm.Bcast([rcounts_N, numprocs, MPI.INT], root=0)
    comm.Bcast([displs_N, numprocs, MPI.INT], root=0)

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

    return 0
