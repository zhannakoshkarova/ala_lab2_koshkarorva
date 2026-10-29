import numpy as np

def find_eig(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    for i in range(len(eigenvalues)):
        lambd = eigenvalues[i]
        v = eigenvectors[:, i]

        if not np.allclose(A @ v, lambd * v):
            raise ValueError(f"Wrong eigenvalue {lambd}")

    return eigenvalues, eigenvectors

