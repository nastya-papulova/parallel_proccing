import numpy as np
import os

matrix_A_size = (20, 30)
vector_x_len = 30

def write_data_info(matrix_A_size=matrix_A_size, vector_x_len=vector_x_len):
    # Writes info about data files in a file.
    os.makedirs('Data', exist_ok=True)
    try:
        with open('Data/data_info.dat', 'w') as file:
            file.write(f"matrix_A_size = {matrix_A_size}\nvector_x_len = {vector_x_len}")
    except FileNotFoundError as e:
        print(f'Error writing to file {file}: {e}')    

def write_matrix_A_data(matrix_A_size=matrix_A_size):   
    # Generate and write a random matrix to a file, one element per line.
    os.makedirs('Data', exist_ok=True)
    M, N = matrix_A_size
    try:
        with open('Data/matrix_A.dat', 'w') as file:
            matrix_A = np.random.random_sample(matrix_A_size)
            file.write("\n".join(map(str, matrix_A.reshape(M*N))))
    except FileNotFoundError as e:
        print(f'Error writing to file {file}: {e}') 

def write_vector_x_data(vector_x_len=vector_x_len): 
    # Generate and write a random vector to a file, one element per line.
    os.makedirs('Data', exist_ok=True) 
    try:
        with open('Data/vector_x.dat', 'w') as file:
            vector_x = np.random.random_sample(vector_x_len)
            file.write("\n".join(map(str, vector_x)))
    except FileNotFoundError as e:
        print(f'Error writing to file {file}: {e}') 

write_data_info(matrix_A_size=matrix_A_size, vector_x_len=vector_x_len)
write_matrix_A_data(matrix_A_size=matrix_A_size)
write_vector_x_data(vector_x_len=vector_x_len)
