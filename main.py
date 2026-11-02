import numpy as np
from matplotlib.image import imread
import matplotlib as plt

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

image_raw = imread("images.jpeg")
print(image_raw.shape)

# image_sum = image_raw.sum(axis=2)
# print(image_sum.shape)
# image_bw = image_sum/image_sum.max()
# print(image_bw.max())

flatten_image = image_raw.flatten()
mean = np.mean(flatten_image)

normalized_image = flatten_image - mean
normalized_image_reshaped = normalized_image.reshape(image_raw.shape)

cov_matrix  = np.cov(normalized_image_reshaped, rowvar=False)

eigenvalues, eigenvectors = find_eig(cov_matrix)
