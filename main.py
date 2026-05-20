from part1 import load_frames, bgr_to_ycbcr, ycbcr_to_bgr
from part3 import encode_video, decode_video
from part4 import write_bin, read_bin
from part5 import visualize_pipeline
import cv2
import os
import time


GOP = 4
Q_factor = 4

frames_bgr = load_frames("frames/")
frames_Y, frames_CbCr = [], []

for frame in frames_bgr:
    Y, Cb, Cr = bgr_to_ycbcr(frame)
    frames_Y.append(Y)
    frames_CbCr.append((Cb, Cr))

print("Encodage...")
start = time.time()
encoded = encode_video(frames_Y, GOP=GOP, Q_factor=Q_factor)
end = time.time()
print("Encoding time:", end - start, "seconds")
for f in encoded:
    f.pop('reconstructed', None)

metadata = {
    'GOP': GOP,
    'Q_factor': Q_factor,
    'num_frames': len(frames_Y),
    'frame_shape': frames_Y[0].shape
}
write_bin(encoded, "output/video.bin", metadata)

print("\nDécodage...")
encoded_loaded, meta = read_bin("output/video.bin")
decoded_Y = decode_video(encoded_loaded, Q_factor=meta['Q_factor'])

os.makedirs("output", exist_ok=True)
for i, Y_rec in enumerate(decoded_Y):
    Cb, Cr = frames_CbCr[i]
    img = ycbcr_to_bgr(Y_rec, Cb, Cr)
    cv2.imwrite(f"output/decoded_frame_{i:03d}.png", img)

print(f"\n{len(decoded_Y)} frames reconstruites dans output/")

print("\nGénération de la visualisation...")
visualize_pipeline(frames_bgr, encoded_loaded, decoded_Y, Q_factor)
