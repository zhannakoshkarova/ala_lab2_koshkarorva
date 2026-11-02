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
print(image_raw.shape)
plt.imshow(image_raw)
plt.show()

image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum/image_sum.max()
print(image_bw.max())
plt.imshow(image_bw, cmap = 'gray')
plt.show()

mean = image_bw.mean(axis=0)

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

def reconstruct_image(k):
    return np.clip((normalized_image @ sorted_eigenvectors[:, :k]) @ (sorted_eigenvectors[:, :k]).T + mean, 0 , 1)

plt.imshow(reconstruct_image(t), cmap = 'gray')
plt.show()

fig, axes = plt.subplots(2,3, figsize = (15,8))
for ax, k in zip(axes.ravel(), [5, 15, 25, 75, 100, 170 ]):
    ax.imshow(reconstruct_image(k), cmap = 'gray')
    ax.set_title(f"Components amount : {k}")
plt.show()


def encrypt_message(message, key_matrix):

    massege_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, massege_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    inv_diag = np.diag(1 / eigenvalues)
    inv_key = eigenvectors @ inv_diag @ np.linalg.inv(eigenvectors)
    message_vector = inv_key @ encrypted_vector
    message_int = np.round(message_vector.real).astype(int)
    message = "".join(chr(i) for i in message_int)

    return message

message = 'Hello, World!'
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted = encrypt_message(message,key_matrix)
print(f"message : {message}")
print(f"encrypted : {encrypted}")
decrypted = decrypt_message(encrypted, key_matrix)
print(f"decrypted : {decrypted}")
