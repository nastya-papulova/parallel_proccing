import re
import time

import numpy as np
from mpi4py import MPI


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

    else:
        raise ValueError("The expected data format was not found in the file.")


def sequential_multiplication(
    matrix_A_size,
    vector_x_len,
    matrix_A_data_filename,
    vector_x_data_filename,
):
    """
    Performs regular (sequential) matrix-vector multiplication.

    Parameters:
    matrix_A_size (tuple): The size of matrix A.
    vector_x_len (int): The length of vector x.
    matrix_A_data_filename (str): The filename containing the data for matrix A.
    vector_x_data_filename (str): The filename containing the data for vector x.

    Returns:
    np.ndarray: The resulting vector after multiplication.
    """

    matrix_A = np.loadtxt(matrix_A_data_filename).reshape(matrix_A_size)
    vector_x = np.loadtxt(vector_x_data_filename)

    b = np.dot(matrix_A, vector_x)

    return b


def parallel_multiplication(
    matrix_A_size,
    vector_x_len,
    matrix_A_data_filename,
    vector_x_data_filename,
):
    """
    Performs parallel matrix-vector multiplication using MPI.

    Parameters:
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

    # Generate arrays rcounts - stores the lengths of parts of the array on each process and displs - stores the indexes of the first elements
    if rank == 0:
        displs = np.empty(numprocs, dtype=np.int32)
        rcounts = np.empty(numprocs, dtype=np.int32)
        displs[0] = 0
        rcounts[0] = 0
        ave, res = divmod(M, numprocs - 1)
        rcounts[1:] = [int(ave + 1 if i <= res else ave) for i in range(1, numprocs)]
        displs[1:] = np.cumsum(rcounts[:-1])
    else:
        rcounts = None
        displs = None

    # Each process gets the number of rows it will handle
    M_part = np.array(0, dtype=np.int32)
    comm.Scatterv(
        [rcounts, np.ones(numprocs), np.arange(numprocs), MPI.INT],
        [M_part, 1, MPI.INT],
        root=0,
    )

    # Allocate space for the part of matrix A each process will handle
    A_part = np.empty((M_part, N), dtype=np.float64)

    if rank == 0:
        matrix_A = np.empty(matrix_A_size, dtype=np.float64)
        matrix_A = np.loadtxt(matrix_A_data_filename).reshape(matrix_A_size)

        # Scatter the parts of matrix A to all processes
        comm.Scatterv(
            [matrix_A, rcounts * N, displs * N, MPI.DOUBLE],
            [A_part, M_part * N, MPI.DOUBLE],
            root=0,
        )

    else:
        comm.Scatterv(
            [None, None, None, None], [A_part, M_part * N, MPI.DOUBLE], root=0
        )

    # Allocate space for the vector x
    vector_x = np.empty(vector_x_len, dtype=np.float64)

    if rank == 0:
        vector_x = np.loadtxt(vector_x_data_filename)

    # Broadcast the vector x to all processes
    comm.Bcast([vector_x, N, MPI.DOUBLE], root=0)

    b_part = np.dot(A_part, vector_x)

    # Each process computes its part of the result
    if rank == 0:
        b = np.empty(M, dtype=np.float64)
    else:
        b = None

    # Gather the parts of the result from all processes
    comm.Gatherv(
        [b_part, rcounts[rank], MPI.DOUBLE], [b, rcounts, displs, MPI.DOUBLE], root=0
    )

    return b


data_info_filename = "Data/data_info.dat"
vector_x_data_filename = "Data/vector_x.dat"
matrix_A_data_filename = "Data/matrix_A.dat"

matrix_A_size, vector_x_len = read_data_info(data_info_filename)

# Initialize MPI
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

start_time = time.time()

b_parallel = parallel_multiplication(
    matrix_A_size,
    vector_x_len,
    matrix_A_data_filename,
    vector_x_data_filename,
)
end_time = time.time()


if rank == 0:
    print(f"Parallel multiplication time: {end_time - start_time} seconds")

    start_time = time.time()

    b_sequential = sequential_multiplication(
        matrix_A_size,
        vector_x_len,
        matrix_A_data_filename,
        vector_x_data_filename,
    )
    end_time = time.time()

    print(f"Sequential multiplication time: {end_time - start_time} seconds")
    print(np.allclose(b_parallel, b_sequential))
