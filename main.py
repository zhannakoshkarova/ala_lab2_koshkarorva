import numpy as np

def find_eig(A):
    eigenvalues, eigenvectors = np.real(np.linalg.eig(A))
    for i in eigenvalues:
        for j in eigenvectors:
            if A*j == i*j:
                continue
            else:
                return "Something went wrong"

    return eigenvalues, eigenvectors

