import os
import re

import numpy as np

matrix_size = (1500, 1000)
vector_len = 1000


def write_data_info(filename, explanatory_line, data_info):
    # Writes or updates if exist info about data files in a file
    os.makedirs("Data", exist_ok=True)

    try:
        with open(filename, "r") as file:
            content = file.readlines()
    except FileNotFoundError:
        content = []

    line_exists = False
    for i, line in enumerate(content):
        if explanatory_line in line:
            content[i] = f"{explanatory_line}{data_info}\n"
            line_exists = True
            break

    if not line_exists:
        content.append(f"{explanatory_line}{data_info}\n")

    with open(filename, "w") as file:
        file.writelines(content)


def write_matrix_data(filename, matrix_size: tuple):
    # Generate and write a random matrix to a file, one element per line
    os.makedirs("Data", exist_ok=True)
    M, N = matrix_size

    try:
        with open(filename, "w") as file:
            matrix = np.random.random_sample(matrix_size)
            file.write("\n".join(map(str, matrix.reshape(M * N))))
    except FileNotFoundError as e:
        print(f"Error writing to file {file}: {e}")


def write_vector_data(filename, vector_len):
    # Generate and write a random vector to a file, one element per line
    os.makedirs("Data", exist_ok=True)
    try:
        with open(filename, "w") as file:
            vector = np.random.random_sample(vector_len)
            file.write("\n".join(map(str, vector)))
    except FileNotFoundError as e:
        print(f"Error writing to file {file}: {e}")


def write_sine_data(matrix_filename, vector_filename, matrix_size):
    # Generates a random matrix and a sine vector, computes their dot product, and writes both to files
    os.makedirs("Data", exist_ok=True)

    M, N = matrix_size
    matrix = np.random.random_sample(matrix_size)
    t = np.linspace(0, 1, N, endpoint=False)
    sin = np.sin(2 * np.pi * N * t)
    vector = np.dot(matrix, sin)

    try:
        with open(matrix_filename, "w") as file:
            file.write("\n".join(map(str, matrix.reshape(M * N))))
    except FileNotFoundError as e:
        print(f"Error writing to file {file}: {e}")

    try:
        with open(vector_filename, "w") as file:
            file.write("\n".join(map(str, vector)))
    except FileNotFoundError as e:
        print(f"Error writing to file {file}: {e}")


"""
write_data_info(
    filename="Data/data_info.dat",
    explanatory_line="vector_y_len=",
    data_info=vector_len,
)
write_vector_data(filename="Data/vector_y.dat", vector_len=vector_len)

write_data_info(
    filename="Data/data_info.dat",
    explanatory_line="matrix_A_size=",
    data_info=matrix_size,
)
write_matrix_data(filename="Data/matrix_A.dat", matrix_size=matrix_size)
write_vector_data(filename="Data/vector_x.dat", vector_len=vector_len)
"""
write_data_info(
    filename="Data/data_info.dat",
    explanatory_line="vector_x_len=",
    data_info=vector_len,
)

write_sine_data(
    matrix_filename="Data/matrix_A.dat",
    vector_filename="Data/vector_x.dat",
    matrix_size=matrix_size,
)
