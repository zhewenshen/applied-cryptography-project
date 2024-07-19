import math
import tenseal as ts
import torch


def compute_coefficients(n):
    coeffs = []
    for i in range(n + 1):
        coeff = math.comb(2*i, i) / (4**i)
        coeffs.append(coeff)
    return coeffs


# Example usage
n = 3  # Change this to the desired n value
coefficients = compute_coefficients(n)

print(coefficients)


def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192,
                         coeff_mod_bit_sizes=[60, 21, 21, 21, 21, 60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context


def decrypt(enc):
    return enc.decrypt().tolist()


context = context()

x = [1]
enc_x = ts.ckks_tensor(context, x)

# print(x)
# print(enc_x.decrypt())

# result = enc_x.polyval(coefficients)
# print(coefficients)
# print(decrypt(result))

y = [2]
enc_y = ts.ckks_tensor(context, y)

# result = enc_y.polyval(coefficients)
# print(decrypt(result))


def new_comp(a, b, n, d):
    x = a - b
    for i in range(d):
        x = x.polyval(compute_coefficients(n))

    return 0.5 * (x + 1)


r = new_comp(enc_x, enc_y, 3, 1)

print(decrypt(r))
