import unittest
from scipy.sparse import random
from scipy import stats
import numpy as np

from sparse import SparseMatrix


N_SAMPLES = 10

class TestSparseMatrix(unittest.TestCase):

    def test_convert(self):
        for _ in range(0, N_SAMPLES):
            rng = np.random.default_rng()
            rvs = stats.poisson(25, loc=10).rvs
            S = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
            sparse_matrix = SparseMatrix.from_array(S.A)
            self.assertTrue(np.allclose(S.A, sparse_matrix.toarray(), rtol=1e-3))
    
    def test_dot(self):
        for _ in range(0, N_SAMPLES):
            rng = np.random.default_rng()
            rvs = stats.poisson(25, loc=10).rvs
            S1 = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
            S2 = random(4, 3, density=0.25, random_state=rng, data_rvs=rvs)
            sparse_matrix1 = SparseMatrix.from_array(S1.A)
            sparse_matrix2 = SparseMatrix.from_array(S2.A)

            numpy_dot = S1.A.dot(S2.A)
            sparse_dot = sparse_matrix1.dot(sparse_matrix2)
            self.assertTrue(np.allclose(numpy_dot, sparse_dot.toarray(), rtol=1e-3))
    
    def test_sum(self):
        for _ in range(0, N_SAMPLES):
            rng = np.random.default_rng()
            rvs = stats.poisson(25, loc=10).rvs
            S1 = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
            S2 = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
            sparse_matrix1 = SparseMatrix.from_array(S1.A)
            sparse_matrix2 = SparseMatrix.from_array(S2.A)

            numpy_sum = S1.A + S2.A
            sparse_sum = sparse_matrix1+sparse_matrix2
            self.assertTrue(np.allclose(numpy_sum, sparse_sum.toarray(), rtol=1e-3))
    
    def test_mul(self):
        for _ in range(0, N_SAMPLES):
            rng = np.random.default_rng()
            rvs = stats.poisson(25, loc=10).rvs
            S1 = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
            scale = np.random.uniform(0, 1)
            sparse_matrix1 = SparseMatrix.from_array(S1.A)

            numpy_mul = S1.A * scale
            sparse_mul = sparse_matrix1 * scale
            self.assertTrue(np.allclose(numpy_mul, sparse_mul.toarray(), rtol=1e-3))

if __name__ == '__main__':
    unittest.main()