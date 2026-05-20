import cv2
import numpy as np
import os

def load_frames(folder_path):

    frames = []
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    for file in files:
        img = cv2.imread(os.path.join(folder_path, file))
        if img is not None:
            frames.append(img)
    print(f"{len(frames)} images chargées.")
    return frames


def bgr_to_ycbcr(frame_bgr):

    frame_ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

    Y, Cr, Cb = cv2.split(frame_ycbcr)

    Cb_sub = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2),interpolation=cv2.INTER_LINEAR)
    Cr_sub = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2),interpolation=cv2.INTER_LINEAR)
    return Y, Cb_sub, Cr_sub
def ycbcr_to_bgr(Y, Cb_sub, Cr_sub):

    h, w = Y.shape

    Cb = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)

    merged = cv2.merge([Y, Cr, Cb])
    frame_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return frame_bgr
