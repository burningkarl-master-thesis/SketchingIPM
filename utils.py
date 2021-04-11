import numpy as np
import scipy.sparse

default_rng = np.random.default_rng()

# The random_sparse_matrix function uses a hack to improve performance. Semantically,
#     row_indices = np.concatenate([rng.choice(m, size=s) for _ in range(n)])
# best describes the intended result. Unfortunately, invoking rng.choice that many
# times takes a very long time. The two options below are faster by about a factor
# of 100.
#
# 1) Generating the row indices using rng.choice(m, size=s * n) comes at the cost of
# possibly defining indices multiple times. This is silently ignored by coo_matrix: The
# last entry with the same index wins. Still, it leads to columns with less than s
# entries.
#     row_indices = rng.choice(m, size=n * s)
#
# 2) An alternative implementation generates the row indices via one call to rng.choice
# and then discards the generated row indices for the columns where these are not
# unique. To guarantee that enough row indices are generated even when some are
# discarded 10% more random entries are generated. For m = 1000, s = 3 the probability
# of a collision is 3/1000 = sC2 / m, so 10% more entries should be plenty.
#     row_indices = rng.choice(m, size=(int(n * 1.1), s))
#     row_indices.sort()
#     row_indices = row_indices[
#         (row_indices[..., 1:] != row_indices[..., :-1]).all(axis=-1)
#     ]
#     row_indices = row_indices[:n]
#     row_indices.shape = (n * s,)


def random_sparse_matrix(
    m: int,
    n: int,
    s: int,
    data: int,
    rng: np.random.Generator = default_rng,
    fast: bool = False,
) -> scipy.sparse.spmatrix:
    """ Generates an m x n matrix with s random entries per column taken from data """
    # For each column sample s random indices
    if fast:
        row_indices = rng.choice(m, size=n * s)
    else:
        row_indices = rng.choice(m, size=(int(n * 1.1), s))
        row_indices.sort()
        row_indices = row_indices[
            (row_indices[..., 1:] != row_indices[..., :-1]).all(axis=-1)
        ]
        row_indices = row_indices[:n]
        row_indices.shape = (n * s,)
    # For each column index i generate i, ..., i (s times)
    # [0, 0, 0, 1, 1, 1, ..., n, n, n] for s=3
    column_indices = np.repeat(np.arange(n), s)
    return scipy.sparse.coo_matrix((data, (row_indices, column_indices)), shape=(m, n))


def gaussian_sketch(
    w: int, n: int, rng: np.random.Generator = default_rng
) -> np.ndarray:
    """ Generates a Gaussian sketching matrix of size w x n"""
    return rng.normal((w, n)) / np.sqrt(w)


def sparse_sketch(
    w: int, n: int, s: int = 3, rng: np.random.Generator = default_rng
) -> scipy.sparse.spmatrix:
    """ Generates a sparse embedding matrix of size w x n with s nonzero entries per column """
    data = rng.choice([-1 / np.sqrt(s), 1 / np.sqrt(s)], size=s * n)
    mat = random_sparse_matrix(w, n, s, data, rng, fast=True)
    return mat.tocsr()


def random_coefficient_matrix(
    m: int, n: int, rng: np.random.Generator = default_rng
) -> np.ndarray:
    """ Generates a random coefficient matrix of size m x n """
    return rng.random((m, n))


def random_sparse_coefficient_matrix(
    m: int, n: int, nnz_per_column: int, rng: np.random.Generator = default_rng
) -> scipy.sparse.spmatrix:
    """ Generates a random sparse coefficient matrix of size m x n with rank m """
    # Randomly scattered nonzero entries
    data = rng.normal(size=nnz_per_column * n)
    mat = random_sparse_matrix(m, n, nnz_per_column, data, rng).todok()
    # One dense row
    mat[0, :] = np.ones((1, n))
    # Nonzero entries on the diagonals
    mat += scipy.sparse.diags(rng.normal(size=m), shape=(m, n))
    return mat.tocsr()


if __name__ == "__main__":
    print("Gaussian sketch:")
    print(gaussian_sketch(5, 10))
    print("Sparse sketch:")
    print(sparse_sketch(5, 10).toarray())
    print("Coefficient matrix:")
    print(random_coefficient_matrix(5, 10))
    print("Sparse coefficient matrix:")
    print(random_sparse_coefficient_matrix(5, 10, nnz_per_column=2).toarray())
