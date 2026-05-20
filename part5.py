import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from part1 import bgr_to_ycbcr, ycbcr_to_bgr
from part2 import dct_2d, quantize, dequantize, idct_2d, Q_MATRIX

os.makedirs("output", exist_ok=True)
def visualize_pipeline(frames_bgr, encoded, decoded_Y, Q_factor=1.0, bin_path="output/video.bin"):

    n = min(5, len(frames_bgr))
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i}", fontsize=11)
        ax.axis('off')
    fig.suptitle("1. Original Frames", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/viz1_original_frames.png", dpi=120, bbox_inches='tight')
    plt.close()

    Y, Cb, Cr = bgr_to_ycbcr(frames_bgr[0])
    Cb_up = cv2.resize(Cb, (Y.shape[1], Y.shape[0]))
    Cr_up = cv2.resize(Cr, (Y.shape[1], Y.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ch, title in zip(axes,
                              [Y, Cb_up, Cr_up],
                              ['Y  (Luminance)', 'Cb  (Blue diff)', 'Cr  (Red diff)']):
        ax.imshow(ch, cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    fig.suptitle("2. YCbCr Color Channels", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/viz2_color_channels.png", dpi=120, bbox_inches='tight')
    plt.close()

    block_raw   = Y[64:72, 64:72].astype(np.float32) - 128
    Q           = Q_MATRIX * Q_factor
    dct_block   = dct_2d(block_raw)
    quant_block = quantize(dct_block, Q).astype(np.float32)
    recon_block = np.clip(idct_2d(dequantize(quant_block, Q)) + 128, 0, 255)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Raw pixels', 'DCT coefficients', 'Quantized', 'Reconstructed']
    datas  = [block_raw + 128, dct_block, quant_block, recon_block]
    cmaps  = ['gray', 'RdBu_r', 'RdBu_r', 'gray']
    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        im = ax.imshow(data, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("3. DCT & Quantization — one 8×8 block", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/viz3_dct_quantization.png", dpi=120, bbox_inches='tight')
    plt.close()

    p_idx = next((i for i, f in enumerate(encoded) if f['type'] == 'P'), 1)
    curr_rgb = cv2.cvtColor(frames_bgr[p_idx], cv2.COLOR_BGR2RGB)
    h_f, w_f = curr_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(curr_rgb)

    mvs = encoded[p_idx]['motion_vectors']
    xs, ys, dxs, dys = [], [], [], []
    for (bx, by, dx, dy) in mvs:
        if abs(dx) + abs(dy) > 0:
            xs.append(bx + 8)
            ys.append(by + 8)
            dxs.append(dx)
            dys.append(dy)

    if xs:
        ax.quiver(xs, ys, dxs, dys,
                  color='red', angles='xy', scale_units='xy',
                  scale=0.25, width=0.003, headwidth=4, headlength=4)

    ax.set_title(f"4. Motion Vectors — P-frame {p_idx}", fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("output/viz4_motion_vectors.png", dpi=120, bbox_inches='tight')
    plt.close()

    Y_curr = bgr_to_ycbcr(frames_bgr[p_idx])[0].astype(np.float32)
    Y_prev = bgr_to_ycbcr(frames_bgr[p_idx - 1])[0].astype(np.float32)
    residual_temporal = Y_curr - Y_prev

    Y_dec = decoded_Y[p_idx].astype(np.float32)
    residual_encoding = Y_curr - Y_dec              

    _, Cb_p, Cr_p = bgr_to_ycbcr(frames_bgr[p_idx])
    recon_bgr = ycbcr_to_bgr(decoded_Y[p_idx], Cb_p, Cr_p)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(cv2.cvtColor(frames_bgr[p_idx], cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original (frame {p_idx})", fontsize=11)
    axes[0].axis('off')

    im1 = axes[1].imshow(residual_temporal, cmap='RdBu_r', vmin=-50, vmax=50)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Temporal residual\n(frame {p_idx} - frame {p_idx-1})", fontsize=11)
    axes[1].axis('off')

    im2 = axes[2].imshow(residual_encoding, cmap='RdBu_r', vmin=-50, vmax=50)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Encoding residual\n(original - reconstructed)", fontsize=11)
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Reconstructed (frame {p_idx})", fontsize=11)
    axes[3].axis('off')

    fig.suptitle("5. Residuals & Reconstruction", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/viz5_residuals.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("Saved: viz1_original_frames.png")
    print("Saved: viz2_color_channels.png")
    print("Saved: viz3_dct_quantization.png")
    print("Saved: viz4_motion_vectors.png")
    print("Saved: viz5_residuals.png")

    Y_ov, Cb_ov, Cr_ov = bgr_to_ycbcr(frames_bgr[0])
    Cb_ov_up = cv2.resize(Cb_ov, (Y_ov.shape[1], Y_ov.shape[0]))
    Cr_ov_up = cv2.resize(Cr_ov, (Y_ov.shape[1], Y_ov.shape[0]))

    n_frames = min(5, len(frames_bgr))

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("MPEG-4 Encoder Pipeline — Full Overview", fontsize=15, fontweight='bold', y=0.98)

    C = max(n_frames, 4)

    for i in range(n_frames):
        ax = fig.add_subplot(5, C, i + 1)
        ax.imshow(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB))
        ftype = encoded[i]['type'] if i < len(encoded) else '?'
        ax.set_title(f"Frame {i}  [{ftype}]", fontsize=9)
        ax.axis('off')
    for j in range(n_frames, C):
        fig.add_subplot(5, C, j + 1).axis('off')

    color_data   = [Y_ov, Cb_ov_up, Cr_ov_up]
    color_titles = ['Y  (Luminance)', 'Cb  (Blue diff)', 'Cr  (Red diff)']
    for k, (ch, title) in enumerate(zip(color_data, color_titles)):
        ax = fig.add_subplot(5, C, C + k + 1)
        ax.imshow(ch, cmap='gray')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    for j in range(3, C):
        fig.add_subplot(5, C, C + j + 1).axis('off')

    dct_stages  = [block_raw + 128, dct_block, quant_block, recon_block]
    dct_titles  = ['Raw pixels', 'DCT coeffs', 'Quantised', 'Reconstructed']
    dct_cmaps   = ['gray', 'RdBu_r', 'RdBu_r', 'gray']
    for k, (data, title, cmap) in enumerate(zip(dct_stages, dct_titles, dct_cmaps)):
        ax = fig.add_subplot(5, C, 2*C + k + 1)
        im = ax.imshow(data, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(4, C):
        fig.add_subplot(5, C, 2*C + j + 1).axis('off')

    ax_mv = fig.add_subplot(5, C, 3*C + 1)
    ax_mv.imshow(cv2.cvtColor(frames_bgr[p_idx], cv2.COLOR_BGR2RGB))
    if xs:
        ax_mv.quiver(xs, ys, dxs, dys,
                     color='red', angles='xy', scale_units='xy',
                     scale=0.25, width=0.003, headwidth=4, headlength=4)
    ax_mv.set_title(f"Motion vectors — P-frame {p_idx}", fontsize=9)
    ax_mv.axis('off')
    for j in range(1, C):
        fig.add_subplot(5, C, 3*C + j + 1).axis('off')

    row5_data   = [
        cv2.cvtColor(frames_bgr[p_idx], cv2.COLOR_BGR2RGB),
        residual_temporal,
        residual_encoding,
        cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB),
    ]
    row5_titles = [
        f"Original (frame {p_idx})",
        f"Temporal residual\n(frame {p_idx} − frame {p_idx-1})",
        "Encoding residual\n(original − reconstructed)",
        f"Reconstructed (frame {p_idx})",
    ]
    row5_cmaps  = [None, 'RdBu_r', 'RdBu_r', None]
    for k, (data, title, cmap) in enumerate(zip(row5_data, row5_titles, row5_cmaps)):
        ax = fig.add_subplot(5, C, 4*C + k + 1)
        if cmap:
            im = ax.imshow(data, cmap=cmap, vmin=-50, vmax=50)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    for j in range(4, C):
        fig.add_subplot(5, C, 4*C + j + 1).axis('off')

    row_labels = [
        "1  Original\nframes",
        "2  Color\nspace",
        "3  DCT &\nQuantisation",
        "4  Motion\nvectors",
        "5  Residuals &\nReconstruction",
    ]
    for row, label in enumerate(row_labels):
        y_pos = 1 - (row + 0.5) / 5
        fig.text(0.01, y_pos, label, va='center', ha='left',
                 fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#eeeeee', edgecolor='none'))

    plt.tight_layout(rect=[0.07, 0, 1, 0.97])
    plt.savefig("output/viz6_pipeline_overview.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: viz6_pipeline_overview.png")
def compute_metrics(frames_bgr, decoded_Y, encoded, bin_path="output/video.bin"):
    
    print("\n===== MÉTRIQUES PAR FRAME =====")
    
    i_count = sum(1 for f in encoded if f['type'] == 'I')
    p_count = sum(1 for f in encoded if f['type'] == 'P')
    print(f"I-frames : {i_count}  |  P-frames : {p_count}  |  Total : {len(encoded)}")
    
    psnr_values = []
    
    for i, (frame_bgr, Y_dec) in enumerate(zip(frames_bgr, decoded_Y)):
        Y_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        Y_rec  = Y_dec.astype(np.float32)
        
        mse = np.mean((Y_orig - Y_rec) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10((255 ** 2) / mse)
        
        psnr_values.append(psnr)
        ftype = encoded[i]['type']
        print(f"  Frame {i:03d} [{ftype}] — PSNR : {psnr:.2f} dB")
    
    print(f"\nPSNR moyen : {np.mean(psnr_values):.2f} dB")
    
    if os.path.exists(bin_path):
        bin_size = os.path.getsize(bin_path)
        print(f"Taille fichier .bin : {bin_size / 1024:.1f} KB")
    
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(psnr_values)), psnr_values, color='orange', linewidth=0.8, label='P-frame')
    
    i_indices = [i for i, f in enumerate(encoded) if f['type'] == 'I']
    i_psnr    = [psnr_values[i] for i in i_indices]
    plt.scatter(i_indices, i_psnr, color='blue', s=20, zorder=5, label='I-frame')
    
    plt.xlabel("Frame index")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR par frame (bleu = I-frame, orange = P-frame)")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("output/viz7_psnr_per_frame.png", dpi=120)
    plt.close()
    print("Saved: viz7_psnr_per_frame.png")
    
    return psnr_values