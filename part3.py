import numpy as np
import cv2
from part2 import encode_iframe, decode_iframe, quantize, dequantize, dct_2d, idct_2d, Q_MATRIX


def motion_estimation(current_block, ref_frame, block_x, block_y, search_window=8):

    h, w = ref_frame.shape

    LDSP = [(0, 0), (0, -2), (2, 0), (0, 2), (-2, 0)]
    SDSP = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]

    curr = current_block.astype(np.float32)

    def get_mse(dx, dy):
        x = block_x + dx
        y = block_y + dy

        if x < 0 or y < 0 or x + 16 > w or y + 16 > h:
            return float('inf')

        ref_block = ref_frame[y:y+16, x:x+16].astype(np.float32)
        diff = curr - ref_block
        return np.mean(diff * diff)

    best_dx, best_dy = 0, 0
    best_mse = get_mse(0, 0)


    improved = True
    while improved:
        improved = False

        for dx, dy in LDSP:
            cand_dx = best_dx + dx
            cand_dy = best_dy + dy

            mse = get_mse(cand_dx, cand_dy)

            if mse < best_mse:
                best_mse = mse
                best_dx, best_dy = cand_dx, cand_dy
                improved = True


    improved = True
    while improved:
        improved = False

        for dx, dy in SDSP:
            cand_dx = best_dx + dx
            cand_dy = best_dy + dy

            mse = get_mse(cand_dx, cand_dy)

            if mse < best_mse:
                best_mse = mse
                best_dx, best_dy = cand_dx, cand_dy
                improved = True

    return best_dx, best_dy
def encode_pframe(current_Y, ref_Y, Q_factor=1.0):
    h, w = current_Y.shape
    Q = Q_MATRIX * Q_factor
    motion_vectors = []
    residual_coeffs = np.zeros((h, w), dtype=np.int32)
    for block_y in range(0, h - 15, 16):
        for block_x in range(0, w - 15, 16):
            current_block = current_Y[block_y:block_y+16, block_x:block_x+16]
            dx, dy = motion_estimation(current_block, ref_Y, block_x, block_y)
            motion_vectors.append((block_x, block_y, dx, dy))
            ref_x = np.clip(block_x + dx, 0, w - 16)
            ref_y = np.clip(block_y + dy, 0, h - 16)
            predicted_block = ref_Y[ref_y:ref_y+16, ref_x:ref_x+16]
            residual = current_block.astype(np.float32) - predicted_block.astype(np.float32)
            for i in range(0, 16, 8):
                for j in range(0, 16, 8):
                    sub = residual[i:i+8, j:j+8]
                    dct_sub = dct_2d(sub)
                    quant_sub = quantize(dct_sub, Q)
                    residual_coeffs[block_y+i:block_y+i+8, block_x+j:block_x+j+8] = quant_sub
    return motion_vectors, residual_coeffs
def decode_pframe(ref_Y, motion_vectors, residual_coeffs, Q_factor=1.0):
    h, w = ref_Y.shape
    Q = Q_MATRIX * Q_factor
    reconstructed = np.zeros((h, w), dtype=np.float32)
    for (block_x, block_y, dx, dy) in motion_vectors:
        ref_x = np.clip(block_x + dx, 0, w - 16)
        ref_y = np.clip(block_y + dy, 0, h - 16)
        predicted_block = ref_Y[ref_y:ref_y+16, ref_x:ref_x+16].astype(np.float32)
        residual = np.zeros((16, 16), dtype=np.float32)
        for i in range(0, 16, 8):
            for j in range(0, 16, 8):
                quant_sub = residual_coeffs[block_y+i:block_y+i+8, block_x+j:block_x+j+8].astype(np.float32)
                dct_sub = dequantize(quant_sub, Q)
                residual[i:i+8, j:j+8] = idct_2d(dct_sub)
        block = predicted_block + residual
        reconstructed[block_y:block_y+16, block_x:block_x+16] = np.clip(block, 0, 255)
    return reconstructed.astype(np.uint8)
def encode_video(frames_Y, GOP=4, Q_factor=1.0):
    encoded = []
    for i, Y in enumerate(frames_Y):
        if i % GOP == 0:
            coeffs = encode_iframe(Y, Q_factor)
            encoded.append({'type': 'I', 'coeffs': coeffs})
            print(f"Frame {i} → I-frame")
        else:
            ref_Y = encoded[-1]['reconstructed']
            mvs, res_coeffs = encode_pframe(Y, ref_Y, Q_factor)
            encoded.append({'type': 'P', 'motion_vectors': mvs, 'residual_coeffs': res_coeffs})
            print(f"Frame {i} → P-frame")
        if encoded[-1]['type'] == 'I':
            encoded[-1]['reconstructed'] = decode_iframe(encoded[-1]['coeffs'], Q_factor)
        else:
            encoded[-1]['reconstructed'] = decode_pframe(
                encoded[-2]['reconstructed'],
                encoded[-1]['motion_vectors'],
                encoded[-1]['residual_coeffs'],
                Q_factor
            )
    return encoded
def decode_video(encoded, Q_factor=1.0):
    decoded_frames = []
    for i, frame_data in enumerate(encoded):
        if frame_data['type'] == 'I':
            Y = decode_iframe(frame_data['coeffs'], Q_factor)
        else:
            ref_Y = decoded_frames[-1]
            Y = decode_pframe(ref_Y, frame_data['motion_vectors'],
                              frame_data['residual_coeffs'], Q_factor)
        decoded_frames.append(Y)
        print(f"Frame {i} décodée ({frame_data['type']}-frame)")
    return decoded_frames
