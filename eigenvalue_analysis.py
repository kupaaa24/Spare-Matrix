import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

# Step 1: Define dimensions and sparsity
n_rows, n_cols = 1000, 1000  # Size of the sparse matrix
density = 0.01  # 1% density

# Step 2: Create a random sparse matrix
sparse_matrix = sparse.random(n_rows, n_cols, density=density, format='csr', dtype=np.float64)

# Step 3: Optionally convert to a symmetric matrix
sparse_matrix = sparse_matrix + sparse_matrix.T
sparse_matrix.setdiag(0)  # Set diagonal to zero

# Print the sparse matrix shape and number of non-zero entries
print("Sparse Matrix Shape:", sparse_matrix.shape)
print("Non-zero entries:", sparse_matrix.nnz)

# Step 4: Calculate eigenvalues and eigenvectors
num_eigenvalues = 5  # Number of eigenvalues to compute
eigenvalues, eigenvectors = eigs(sparse_matrix, k=num_eigenvalues)

# Print the results
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Step 5: Plotting the eigenvalues in a bar graph
plt.figure(figsize=(12, 6))

# Bar for real part
plt.subplot(1, 2, 1)
plt.bar(range(num_eigenvalues), np.real(eigenvalues), color='b', alpha=0.7)
plt.title('Real Part of Eigenvalues')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Value')
plt.xticks(range(num_eigenvalues), [f'λ{i+1}' for i in range(num_eigenvalues)])
plt.grid()

# Bar for imaginary part
plt.subplot(1, 2, 2)
plt.bar(range(num_eigenvalues), np.imag(eigenvalues), color='r', alpha=0.7)
plt.title('Imaginary Part of Eigenvalues')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Value')
plt.xticks(range(num_eigenvalues), [f'λ{i+1}' for i in range(num_eigenvalues)])
plt.grid()

plt.tight_layout()
plt.show()
