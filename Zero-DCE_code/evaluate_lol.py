"""Evaluate a Zero-DCE checkpoint on the LOL test split (eval15).

Computes PSNR and SSIM against the normal-light ground truth (scikit-image),
saves the enhanced images, per-image scores (per_image.csv) and the means (summary.json).

Example:
    python evaluate_lol.py --weights snapshots/Epoch99.pth --lol_dir data/LOL/eval15
"""
import argparse
import csv
import json
import os
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import model


def get_device():
	if torch.cuda.is_available():
		return torch.device('cuda')
	if torch.backends.mps.is_available():
		return torch.device('mps')
	return torch.device('cpu')


def load_rgb(path):
	return np.asarray(Image.open(path).convert('RGB'))


def to_tensor(img_uint8):
	return torch.from_numpy(img_uint8 / 255.0).float().permute(2, 0, 1).unsqueeze(0)


def main(config):
	device = get_device()
	print("Using device:", device)

	DCE_net = model.enhance_net_nopool().to(device).eval()
	DCE_net.load_state_dict(torch.load(config.weights, map_location=device))

	low_paths = sorted(glob(os.path.join(config.lol_dir, 'low', '*.png')))
	if not low_paths:
		raise SystemExit(f"No images found in {config.lol_dir}/low")
	os.makedirs(config.out_dir, exist_ok=True)

	rows = []
	with torch.no_grad():
		for low_path in low_paths:
			name = os.path.basename(low_path)
			low = load_rgb(low_path)
			gt = load_rgb(os.path.join(config.lol_dir, 'high', name))

			_, enhanced, _ = DCE_net(to_tensor(low).to(device))

			enhanced = enhanced.clamp(0, 1).cpu()
			torchvision.utils.save_image(enhanced, os.path.join(config.out_dir, name))
			out = (enhanced[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

			psnr = peak_signal_noise_ratio(gt, out, data_range=255)
			ssim = structural_similarity(gt, out, data_range=255, channel_axis=2)

			rows.append({'image': name, 'psnr': psnr, 'ssim': ssim})
			print(f"{name}: PSNR {psnr:.2f}  SSIM {ssim:.4f}")

	summary = {
		'weights': config.weights,
		'n_images': len(rows),
		'psnr': float(np.mean([r['psnr'] for r in rows])),
		'ssim': float(np.mean([r['ssim'] for r in rows])),
	}

	with open(os.path.join(config.out_dir, 'per_image.csv'), 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=rows[0].keys())
		writer.writeheader()
		writer.writerows(rows)
	with open(os.path.join(config.out_dir, 'summary.json'), 'w') as f:
		json.dump(summary, f, indent=2)

	print("\n=== Summary ===")
	for k, v in summary.items():
		print(f"{k:>15}: {v}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', type=str, default='snapshots/Epoch99.pth')
	parser.add_argument('--lol_dir', type=str, default='data/LOL/eval15')
	parser.add_argument('--out_dir', type=str, default='results/authors_pretrained')
	config = parser.parse_args()
	main(config)
