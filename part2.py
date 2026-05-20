import numpy as np
import cv2


Q_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)
def dct_2d(block):
    return cv2.dct(block.astype(np.float32))
def idct_2d(block):
    return cv2.idct(block.astype(np.float32))
def quantize(dct_block, Q):
    return np.round(dct_block / Q).astype(np.int32)
def dequantize(quant_block, Q):
    return (quant_block * Q).astype(np.float32)
def encode_iframe(Y_channel, Q_factor=1.0):
    h, w = Y_channel.shape
    Q = Q_MATRIX * Q_factor

    coeffs = np.zeros((h, w), dtype=np.int32)
    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = Y_channel[i:i+8, j:j+8].astype(np.float32) - 128
            dct_block = dct_2d(block)
            quant_block = quantize(dct_block, Q)
            coeffs[i:i+8, j:j+8] = quant_block
    return coeffs
def decode_iframe(coeffs, Q_factor=1.0):
    h, w = coeffs.shape
    Q = Q_MATRIX * Q_factor
    Y_reconstructed = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            quant_block = coeffs[i:i+8, j:j+8].astype(np.float32)
            dct_block = dequantize(quant_block, Q)
            block = idct_2d(dct_block) + 128
            Y_reconstructed[i:i+8, j:j+8] = np.clip(block, 0, 255)
    return Y_reconstructed.astype(np.uint8)
