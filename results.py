import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

gt_folder = "image_super_resolution_dataset/scaling_4x/test/HR"
pred_folder = "saved"

image_files = sorted(os.listdir(gt_folder))

psnr_scores = []
ssim_scores = []

for filename in image_files:
    gt_path = os.path.join(gt_folder, filename)
    pred_path = os.path.join(pred_folder, filename)

    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)

    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    score = psnr(gt, pred)
    psnr_scores.append(score)
    print(f"{filename}: PSNR = {score:.2f} dB")

    score = ssim(gt, pred, channel_axis=2)
    ssim_scores.append(score)
    print(f"{filename}: SSIM = {score:.4f}")

avg_psnr = sum(psnr_scores) / len(psnr_scores)
print(f"\nAverage PSNR over {len(psnr_scores)} images: {avg_psnr:.2f} dB")

avg_ssim = sum(ssim_scores) / len(ssim_scores)
print(f"\nAverage SSIM over {len(ssim_scores)} images: {avg_ssim:.2f}")


plt.figure(figsize=(20, 5))
plt.plot(psnr_scores, marker='o', label='PSNR')
plt.xticks(range(len(psnr_scores)))
plt.title('PSNR per Image')
plt.xlabel('Image Index')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.legend()


plt.figure(figsize=(20, 5))
plt.plot(ssim_scores, marker='s', color='orange', label='SSIM')
plt.xticks(range(len(ssim_scores)))
plt.title('SSIM per Image')
plt.xlabel('Image Index')
plt.ylabel('SSIM')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
