import numpy as np
from scipy.sparse import dok_matrix

rng = np.random.default_rng()


def gaussian_sketch(w: int, m: int) -> np.ndarray:
    return rng.random((w, m)) / np.sqrt(w)


def sparse_sketch(w: int, m: int, s: int = 3) -> dok_matrix:
    # Initialize matrix with zeros
    mat = dok_matrix((w, m))
    # Randomly choose s nonzero indices per column
    idx = rng.random(mat.shape).argsort(axis=0)[:s]
    # Set the entries to +1/-1
    mat[idx, np.arange(mat.shape[1])] = rng.choice([-1, 1], size=idx.shape)
    # Multiply by 1/sqrt(s)
    return mat / np.sqrt(s)


if __name__ == "__main__":
    print('Gaussian sketch:')
    print(gaussian_sketch(5, 10))
    print('Sparse sketch:')
    print(sparse_sketch(5, 10).toarray())
