import re
import time

import numpy as np
from mpi4py import MPI


def read_data_info(filename):
    """
    Reads  the length of vectors x and y from a data file.

    Parameters:
    filename (str): The path to the data file.

    Returns:
    int: length of vectors (same for both).

    Raises:
    ValueError: Two vectors cannot be multiplied due to incorrect dimensions.
    ValueError: If the expected data format is not found in the file.
    """
    with open(filename, "r") as file:
        content = file.read()

    vector_x_len_match = re.search(r"vector_x_len\s*=\s*(\d+)", content)
    vector_y_len_match = re.search(r"vector_y_len\s*=\s*(\d+)", content)

    if vector_x_len_match and vector_y_len_match:
        vector_x_len = int(vector_x_len_match.group(1))
        vector_y_len = int(vector_y_len_match.group(1))
        if vector_x_len != vector_y_len:
            raise ValueError(
                "Vectors cannot be multiplied due to incorrect dimensions."
            )

        return vector_x_len

    else:
        raise ValueError("The expected data format was not found in the file.")


def sequential_dot_product(vector_len, vector_x_data_filename, vector_y_data_filename):
    """
    Performs regular (sequential) dot product of vectors.

    Parameters:
    vector_len (int): The length of vectors.
    vector_x_data_filename (str): The filename containing the data for vector x.
    vector_y_data_filename (str): The filename containing the data for vector y.

    Returns:
    np.ndarray: The resulting vector after dot product.
    """

    vector_x = np.loadtxt(vector_x_data_filename)
    vector_y = np.loadtxt(vector_y_data_filename)

    ScalP = np.dot(vector_x, vector_y)

    return ScalP


def parallel_dot_product(
    comm,
    numprocs,
    rank,
    vector_len,
    vector_x_data_filename,
    vector_y_data_filename,
):
    """
    Performs parallel dot product of vectors using MPI.

    Parameters:
    comm (MPI.Comm): The MPI communicator.
    numprocs (int): The number of processes.
    rank (int): The rank of the current process.
    vector_len (int): The length of vectors.
    vector_x_data_filename (str): The filename containing the data for vector x.
    vector_y_data_filename (str): The filename containing the data for vector y.

    Returns:
    np.ndarray: TThe resulting vector after dot product (only for the root process).
    """

    if rank == 0:
        M = vector_len
        vector_x = np.empty(M, dtype=np.float64)
        vector_y = np.empty(M, dtype=np.float64)

        vector_x = np.loadtxt(vector_x_data_filename)
        vector_y = np.loadtxt(vector_y_data_filename)
    else:
        vector_x = None
        vector_y = None

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
    x_part = np.empty(M_part, dtype=np.float64)
    y_part = np.empty(M_part, dtype=np.float64)
    comm.Scatterv(
        [vector_x, rcounts, displs, MPI.DOUBLE],
        [x_part, M_part, MPI.DOUBLE],
        root=0,
    )
    comm.Scatterv(
        [vector_y, rcounts, displs, MPI.DOUBLE],
        [y_part, M_part, MPI.DOUBLE],
        root=0,
    )

    ScalP_temp = np.empty(1, dtype=np.float64)
    ScalP_temp[0] = np.dot(x_part, y_part)

    ScalP = np.array(0, dtype=np.float64)

    comm.Reduce([ScalP_temp, 1, MPI.DOUBLE], [ScalP, 1, MPI.DOUBLE], op=MPI.SUM, root=0)

    return ScalP


data_info_filename = "Data/data_info.dat"
vector_x_data_filename = "Data/vector_x.dat"
vector_y_data_filename = "Data/vector_y.dat"


vector_len = read_data_info(data_info_filename)

# Initialize MPI
comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

start_time = time.time()

b_parallel = parallel_dot_product(
    comm,
    numprocs,
    rank,
    vector_len=vector_len,
    vector_x_data_filename=vector_x_data_filename,
    vector_y_data_filename=vector_y_data_filename,
)
end_time = time.time()

if rank == 0:
    print(f"Parallel dot product time: {end_time - start_time} seconds")


start_time = time.time()

b_sequential = sequential_dot_product(
    vector_len=vector_len,
    vector_x_data_filename=vector_x_data_filename,
    vector_y_data_filename=vector_y_data_filename,
)
end_time = time.time()

if rank == 0:
    print(f"Sequential multiplication time: {end_time - start_time} seconds")
    print(np.allclose(b_parallel, b_sequential))
