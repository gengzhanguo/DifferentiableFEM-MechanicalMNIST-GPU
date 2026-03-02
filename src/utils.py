import torch
import numpy as np
import scipy
from scipy.sparse import csc_matrix

# Global device and dtype configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

def scipy_sparse_to_torch_tensor(scipy_matrix: scipy.sparse.spmatrix) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a PyTorch tensor."""
    coo = scipy_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to_dense()

def sparse_coo_to_csc(sparse_coo_tensor: torch.Tensor) -> csc_matrix:
    """Convert a PyTorch sparse_coo_tensor to a SciPy csc_matrix."""
    if not sparse_coo_tensor.is_sparse: # Using .is_sparse as per PyTorch conventions
        raise ValueError("Input tensor must be a sparse tensor.")

    indices = sparse_coo_tensor.indices().cpu().numpy()
    values = sparse_coo_tensor.values().cpu().numpy()
    shape = sparse_coo_tensor.shape

    return csc_matrix((values, (indices[0], indices[1])), shape=shape)
