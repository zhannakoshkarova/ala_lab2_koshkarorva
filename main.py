import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

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
# print(image_raw.shape)

image_sum = image_raw.sum(axis=2)
# print(image_sum.shape)
image_bw = image_sum/image_sum.max()
# print(image_bw.max())

mean = np.mean(image_bw)

normalized_image = image_bw - mean
cov_matrix  = np.cov(normalized_image, rowvar=False)

eigenvalues, eigenvectors = find_eig(cov_matrix)

sorted_idx = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = np.clip(eigenvalues[sorted_idx], 0, None)
sorted_eigenvectors = eigenvectors[:, sorted_idx]

cumulative = np.cumsum(sorted_eigenvalues)/ sorted_eigenvalues.sum()
t = np.argmax(cumulative >= 0.95) + 1
print(f"Components for 95%: {t}")

plt.plot(cumulative * 100)
plt.axhline(95, color = 'k', linestyle = '--')
plt.axvline(t, color = 'k', linestyle = '--')
plt.xlabel('principal components')
plt.ylabel('cumulative explained variance (%)')
plt.show()

