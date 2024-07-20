import numpy as np
from Pyfhel import Pyfhel
from tqdm import tqdm

def encrypt_vector(HE, vector):
    return [HE.encrypt(v) for v in vector]

def encrypt_matrix(HE, matrix):
    return [[HE.encrypt(val) for val in row] for row in matrix]

def dot_product(HE, enc_vec1, enc_vec2):
    if len(enc_vec1) != len(enc_vec2):
        raise ValueError("Vectors must have the same length")
    
    products = [v1 * v2 for v1, v2 in zip(enc_vec1, enc_vec2)]
    result = products[0]
    for p in products[1:]:
        result += p
    
    return result

def matrix_product(HE, enc_mat1, enc_mat2):
    rows1 = len(enc_mat1)
    cols1 = len(enc_mat1[0])
    rows2 = len(enc_mat2)
    cols2 = len(enc_mat2[0])
    
    if cols1 != rows2:
        raise ValueError("Matrix dimensions are not compatible for multiplication")
    
    result = [[None for _ in range(cols2)] for _ in range(rows1)]
    
    for i in range(rows1):
        for j in range(cols2):
            dot_prod = dot_product(HE, enc_mat1[i], [row[j] for row in enc_mat2])
            result[i][j] = dot_prod
    
    return result

def matrix_scalar_mul(HE, enc_matrix, scalar):
    rows = len(enc_matrix)
    cols = len(enc_matrix[0])
    
    result = [[None for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = enc_matrix[i][j] * HE.encrypt(scalar)
    
    return result

def matrix_subtraction(HE, enc_mat1, enc_mat2):
    if len(enc_mat1) != len(enc_mat2) or len(enc_mat1[0]) != len(enc_mat2[0]):
        raise ValueError("Matrices must have the same dimensions")
    
    rows = len(enc_mat1)
    cols = len(enc_mat1[0])
    
    result = [[None for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = enc_mat1[i][j] - enc_mat2[i][j]
    
    return result

def matrix_addition(HE, enc_mat1, enc_mat2):
    if len(enc_mat1) != len(enc_mat2) or len(enc_mat1[0]) != len(enc_mat2[0]):
        raise ValueError("Matrices must have the same dimensions")
    
    rows = len(enc_mat1)
    cols = len(enc_mat1[0])
    
    result = [[None for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = enc_mat1[i][j] + enc_mat2[i][j]
    
    return result

## Matrix scalar addition to each element
def matrix_scalar_addition(HE, enc_mat1, scalar):
    rows = len(enc_mat1)
    cols = len(enc_mat1[0])
    
    result = [[None for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = enc_mat1[i][j] + HE.encrypt(scalar)
    
    return result

def matrix_square(HE, enc_matrix):
    rows = len(enc_matrix)
    cols = len(enc_matrix[0])
    
    if rows != cols:
        raise ValueError("Matrix must be square")
    
    return matrix_product(HE, enc_matrix, enc_matrix)


HE = Pyfhel()
ckks_params = {
    'scheme': 'CKKS',
    'n': 2**14,
    'scale': 2**30,
    'qi_sizes': [60, 30, 30, 30, 60]
}
HE.contextGen(**ckks_params)
HE.keyGen()
HE.relinKeyGen()

mat1 = np.array([[1.0, 2.0], [3.0, 4.0]])
mat2 = np.array([[5.0, 6.0], [7.0, 8.0]])
scalar = 2.0
vec1 = np.array([1.0, 2.0, 3.0])
vec2 = np.array([4.0, 5.0, 6.0])

enc_mat1 = encrypt_matrix(HE, mat1)
enc_mat2 = encrypt_matrix(HE, mat2)
enc_vec1 = encrypt_vector(HE, vec1)
enc_vec2 = encrypt_vector(HE, vec2)

enc_result_dot = dot_product(HE, enc_vec1, enc_vec2)
result_dot = HE.decryptFrac(enc_result_dot)[0]
print("Dot product result:")
print(np.round(result_dot, decimals=10))
print("Expected result:")
print(np.dot(vec1, vec2))

enc_result_matmul = matrix_product(HE, enc_mat1, enc_mat2)
result_matmul = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result_matmul]
print("\nMatrix product result:")
print(np.round(result_matmul, decimals=10))
print("Expected result:")
print(np.dot(mat1, mat2))

enc_result_scalar = matrix_scalar_mul(HE, enc_mat1, scalar)
result_scalar = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result_scalar]
print("\nMatrix-scalar multiplication result:")
print(np.round(result_scalar, decimals=10))
print("Expected result:")
print(mat1 * scalar)

enc_result_sub = matrix_subtraction(HE, enc_mat1, enc_mat2)
result_sub = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result_sub]
print("\nMatrix subtraction result:")
print(np.round(result_sub, decimals=10))
print("Expected result:")
print(mat1 - mat2)

enc_result_add = matrix_addition(HE, enc_mat1, enc_mat2)
result_add = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result_add]
print("\nMatrix addition result:")
print(np.round(result_add, decimals=10))
print("Expected result:")
print(mat1 + mat2)

enc_result_square = matrix_square(HE, enc_mat1)
result_square = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result_square]
print("\nMatrix square result:")
print(np.round(result_square, decimals=10))
print("Expected result:")
print(np.dot(mat1, mat1))

def run_all_tests(HE, max_size=4):
    print("AUTO TESTING...")

    def test_dot_product():
        for n in tqdm(range(1, max_size + 1), desc="Dot product"):
            vec1 = np.random.rand(n)
            vec2 = np.random.rand(n)
            enc_vec1 = encrypt_vector(HE, vec1)
            enc_vec2 = encrypt_vector(HE, vec2)
            enc_result = dot_product(HE, enc_vec1, enc_vec2)
            result = HE.decryptFrac(enc_result)[0]
            expected = np.dot(vec1, vec2)
            if not np.isclose(result, expected, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Dot product test failed for vectors of size {n}")

    def test_matrix_product():
        total = max_size * max_size * max_size
        with tqdm(total=total, desc="Matrix product") as pbar:
            for m in range(1, max_size + 1):
                for n in range(1, max_size + 1):
                    for p in range(1, max_size + 1):
                        mat1 = np.random.rand(m, n)
                        mat2 = np.random.rand(n, p)
                        enc_mat1 = encrypt_matrix(HE, mat1)
                        enc_mat2 = encrypt_matrix(HE, mat2)
                        enc_result = matrix_product(HE, enc_mat1, enc_mat2)
                        result = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result]
                        expected = np.dot(mat1, mat2)
                        if not np.allclose(result, expected, rtol=1e-3, atol=1e-3):
                            raise ValueError(f"Matrix product test failed for matrices of size ({m},{n}) and ({n},{p})")
                        pbar.update(1)

    def test_matrix_scalar_mul():
        total = max_size * max_size
        with tqdm(total=total, desc="Matrix-scalar multiplication") as pbar:
            for m in range(1, max_size + 1):
                for n in range(1, max_size + 1):
                    mat = np.random.rand(m, n)
                    scalar = np.random.rand()
                    enc_mat = encrypt_matrix(HE, mat)
                    enc_result = matrix_scalar_mul(HE, enc_mat, scalar)
                    result = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result]
                    expected = mat * scalar
                    if not np.allclose(result, expected, rtol=1e-3, atol=1e-3):
                        raise ValueError(f"Matrix-scalar multiplication test failed for matrix of size ({m},{n})")
                    pbar.update(1)

    def test_matrix_addition():
        total = max_size * max_size
        with tqdm(total=total, desc="Matrix addition") as pbar:
            for m in range(1, max_size + 1):
                for n in range(1, max_size + 1):
                    mat1 = np.random.rand(m, n)
                    mat2 = np.random.rand(m, n)
                    enc_mat1 = encrypt_matrix(HE, mat1)
                    enc_mat2 = encrypt_matrix(HE, mat2)
                    enc_result = matrix_addition(HE, enc_mat1, enc_mat2)
                    result = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result]
                    expected = mat1 + mat2
                    if not np.allclose(result, expected, rtol=1e-3, atol=1e-3):
                        raise ValueError(f"Matrix addition test failed for matrices of size ({m},{n})")
                    pbar.update(1)

    def test_matrix_subtraction():
        total = max_size * max_size
        with tqdm(total=total, desc="Matrix subtraction") as pbar:
            for m in range(1, max_size + 1):
                for n in range(1, max_size + 1):
                    mat1 = np.random.rand(m, n)
                    mat2 = np.random.rand(m, n)
                    enc_mat1 = encrypt_matrix(HE, mat1)
                    enc_mat2 = encrypt_matrix(HE, mat2)
                    enc_result = matrix_subtraction(HE, enc_mat1, enc_mat2)
                    result = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result]
                    expected = mat1 - mat2
                    if not np.allclose(result, expected, rtol=1e-3, atol=1e-3):
                        raise ValueError(f"Matrix subtraction test failed for matrices of size ({m},{n})")
                    pbar.update(1)

    def test_matrix_square():
        for n in tqdm(range(1, max_size + 1), desc="Matrix square"):
            mat = np.random.rand(n, n)
            enc_mat = encrypt_matrix(HE, mat)
            enc_result = matrix_square(HE, enc_mat)
            result = [[HE.decryptFrac(val)[0] for val in row] for row in enc_result]
            expected = np.dot(mat, mat)
            if not np.allclose(result, expected, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Matrix square test failed for matrix of size ({n},{n})")

    tests = [
        test_dot_product,
        test_matrix_product,
        test_matrix_scalar_mul,
        test_matrix_addition,
        test_matrix_subtraction,
        test_matrix_square
    ]

    for test in tests:
        test()

    print("All tests passed successfully!")

run_all_tests(HE)
