import numpy as np
import scipy.sparse

rng = np.random.default_rng()


def gaussian_sketch(w: int, m: int) -> np.ndarray:
    return rng.random((w, m)) / np.sqrt(w)


def sparse_sketch(w: int, m: int, s: int = 3) -> scipy.sparse.spmatrix:
    # Initialize matrix with zeros
    mat = scipy.sparse.dok_matrix((w, m))
    # Randomly choose s nonzero indices per column
    idx = rng.random(mat.shape).argsort(axis=0)[:s]
    # Set the entries to +1/-1
    mat[idx, np.arange(mat.shape[1])] = rng.choice([-1, 1], size=idx.shape)
    # Multiply by 1/sqrt(s)
    return (mat / np.sqrt(s)).tocoo()


def random_coefficient_matrix(m: int, n: int) -> np.ndarray:
    return rng.random((m, n))


def random_sparse_coefficient_matrix(m: int, n: int, density: float = 0.01) -> scipy.sparse.spmatrix:
    return scipy.sparse.random(m, n, density=density) + scipy.sparse.diags(rng.random(m), shape=(m, n))


if __name__ == "__main__":
    print('Gaussian sketch:')
    print(gaussian_sketch(5, 10))
    print('Sparse sketch:')
    print(sparse_sketch(5, 10).toarray())
    print('Coefficient matrix:')
    print(random_coefficient_matrix(5, 10))
    print('Sparse coefficient matrix:')
    print(random_sparse_coefficient_matrix(5, 10, density=0.5).toarray())
