import numpy as np
import scipy.sparse

rng = np.random.default_rng()


def gaussian_sketch(w: int, n: int) -> np.ndarray:
    """ Generates a Gaussian sketching matrix of size w x n"""
    return rng.random((w, n)) / np.sqrt(w)


def sparse_sketch(w: int, n: int, s: int = 3) -> scipy.sparse.spmatrix:
    """ Generates a sparse embedding matrix of size w x n with s nonzero entries per column """
    data = rng.choice([-1 / np.sqrt(s), 1 / np.sqrt(s)], size=s * n)
    row_indices = np.hstack([rng.choice(w, size=s, replace=False) for i in range(n)])
    column_indices = np.repeat(np.arange(n), s)  # [0, 0, 0, 1, 1, 1, ..., n, n, n] for s=3
    mat = scipy.sparse.coo_matrix((data, (row_indices, column_indices)), shape=(w, n))
    return mat.tocsr()


def random_coefficient_matrix(m: int, n: int) -> np.ndarray:
    """ Generates a random coefficient matrix of size m x n """
    return rng.random((m, n))


def random_sparse_coefficient_matrix(m: int, n: int, density: float) -> scipy.sparse.spmatrix:
    """ Generates a random sparse coefficient matrix of size m x n with rank m """
    mat = scipy.sparse.dok_matrix((m, n))
    # One dense row
    mat[0, :] = np.ones((1, n))
    # Nonzero entries on the diagonals
    mat += scipy.sparse.diags(rng.random(m), shape=(m, n))
    # Randomly scattered nonzero entries
    mat += scipy.sparse.random(m, n, density=density)
    return mat.tocsr()


if __name__ == "__main__":
    print('Gaussian sketch:')
    print(gaussian_sketch(5, 10))
    print('Sparse sketch:')
    print(sparse_sketch(5, 10).toarray())
    print('Coefficient matrix:')
    print(random_coefficient_matrix(5, 10))
    print('Sparse coefficient matrix:')
    print(random_sparse_coefficient_matrix(5, 10, density=0.5).toarray())
