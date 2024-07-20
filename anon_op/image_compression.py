import numpy as np
from PIL import Image
from Pyfhel import Pyfhel
from main import encrypt_matrix,matrix_scalar_mul,matrix_scalar_addition
from tqdm import tqdm
from PIL import Image

def arithmetic_quantization(HE,enc_mat_image, levels):
    """
    Apply arithmetic quantization to an image without using comparisons.

    Parameters:
    image (numpy.ndarray): The input image.
    levels (int): The number of quantization levels.

    Returns:
    numpy.ndarray: The quantized image.
    """
    # Calculate interval size
    interval_size = 256 // levels
    enc_mat_image = matrix_scalar_mul(HE, enc_mat_image, (1/interval_size))
    enc_mat_image = matrix_scalar_mul(HE, enc_mat_image, interval_size)
    #enc_mat_image = matrix_scalar_addition(HE, enc_mat_image, (interval_size/2))


    return enc_mat_image

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
# Load the image
## As there don't exist many images small enough for the program to use I resize an existing image, change this value at you leisure
image = Image.open('badcert.jpg').resize((32, 32))
image = np.array(image)
enc_mat_image = encrypt_matrix(HE, image)

# Apply quantization
levels = 8
quantized_image = arithmetic_quantization(HE,enc_mat_image, levels)
quantized_image = [[HE.decryptFrac(val)[0] for val in row] for row in quantized_image]
image = np.round(quantized_image,decimals=10)
# Convert back to an image and save
quantized_image = Image.fromarray(image.astype('uint8'))
quantized_image.save('quantized_image_8_levels_arithmetic.jpg')

