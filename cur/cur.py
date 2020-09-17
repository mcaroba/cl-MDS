import numpy as np
from typing import Tuple
Matrix = np.ndarray
Vector = np.ndarray

def cur_decomposition(M: Matrix, r: int) -> Tuple[Matrix, Matrix, Matrix]:
    """CUR decomposition as described in mmds.org page 406.
    Args:
        M: np.ndarray [M, N], matrix to decompose
        r: int, number of rows to sample
    Return:
        C: Column Matrix: [M, r] (sparse)
        U: U Matrix: [r, r] (dense)
        R: Row Matrix: [r, N] (sparse)
    """
    m, n = M.shape
    r_probs, c_probs = probabilities(M)
    C, c_idx = select_C(M, r, c_probs)
    R, r_idx = select_R(M, r, r_probs)
    U = make_U(M, c_idx, r_idx)
    return C, U, R, r_idx

def probabilities(M: Matrix) -> Tuple[Vector, Vector]:
    """Define probabilities to sample rows and columns by,"""
    squared = np.square(M)
    row_sum = np.sum(squared, 1)
    col_sum = np.sum(squared, 0)
    denom = np.sum(row_sum)
    row_probs = row_sum / denom
    col_probs = col_sum / denom
    return row_probs, col_probs

def select_C(M, r, probs):
    return select_part(M, r, probs, 1)

def select_R(M, r, probs):
    return select_part(M, r, probs, 0)

def select_part(M: Matrix, r: int, probs: Vector, axis: int) -> Tuple[Vector, Vector]:
    """Sample r rows or columns from M according to probs.
    Scale rows by the expected number of times they will be drawn.
    """
    size = M.shape[axis]
    idx = np.random.choice(size, size=r, p=probs)
    selected = np.take(M, idx, axis)
    scale = probs[idx]
    scale = np.sqrt(scale * r)
    scale = np.expand_dims(scale, axis - 1)
    return selected / scale, idx

def make_U(M: Matrix, c: Vector, r: Vector) -> Matrix:
    W = select_W(M, c, r)
    x, e, y = np.linalg.svd(W)
    inv_e = psuedo_inverse(e)
    U = y.T @ np.diag(np.square(inv_e)) @ x.T
    return U

def select_W(M: Matrix, c: Vector, r: Vector) -> Vector:
    return M[r, :][:, c]

def psuedo_inverse(sigma: Vector) -> Vector:
    zeros = sigma == 0
    # Get mask where 0's in sigma are 1
    num = (zeros) == 0
    # Replace zero with one
    denum = zeros + sigma
    # do inverse because 0 is now 0/1
    return num / denum
