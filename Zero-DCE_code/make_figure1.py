"""Figure 1: our reproduced Zero-DCE (seed 42, epoch 99) on three LOL-v1 test images.

Columns: low-light input, our output (red box = zoomed region), ground truth, 5x zoom.
Run from Zero-DCE_code/ after evaluating the seed-42 checkpoint:

    python make_figure1.py
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# (image, crop box (x, y, w, h) in the 600x400 frame)
EXAMPLES = [('179.png', (330, 40, 120, 80)), ('23.png', (40, 120, 120, 80)), ('665.png', (300, 230, 120, 80))]


def main(config):
	scores = {r['image']: r for r in csv.DictReader(open(os.path.join(config.results_dir, 'per_image.csv')))}

	fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(12, 2.35 * len(EXAMPLES)))
	for row, (name, (x, y, w, h)) in enumerate(EXAMPLES):
		low = Image.open(os.path.join(config.lol_dir, 'low', name))
		# Images live in results_dir/images (this repo) or directly in results_dir (fresh evaluate_lol.py output).
		img_dir = os.path.join(config.results_dir, 'images')
		out = Image.open(os.path.join(img_dir if os.path.isdir(img_dir) else config.results_dir, name))
		gt = Image.open(os.path.join(config.lol_dir, 'high', name))
		crop = out.crop((x, y, x + w, y + h)).resize((w * 5, h * 5), Image.NEAREST)
		s = scores[name]
		panels = [(low, 'Low-light input'),
		          (out, f'Our Zero-DCE  (PSNR {float(s["psnr"]):.2f}, SSIM {float(s["ssim"]):.3f})'),
		          (gt, 'Ground truth'),
		          (crop, 'Our Zero-DCE, zoomed crop')]
		for col, (img, title) in enumerate(panels):
			ax = axes[row, col]
			ax.imshow(np.asarray(img))
			ax.set_xticks([]); ax.set_yticks([])
			if row == 0 or col == 1:
				ax.set_title(title, fontsize=9)
			if col == 1:
				ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec='red', lw=1.2))
		axes[row, 0].set_ylabel(name.replace('.png', ''), fontsize=10)
	plt.tight_layout()
	os.makedirs(config.out_dir, exist_ok=True)
	plt.savefig(os.path.join(config.out_dir, 'fig1_reproduction.pdf'), bbox_inches='tight')
	plt.savefig(os.path.join(config.out_dir, 'fig1_reproduction.png'), dpi=150, bbox_inches='tight')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lol_dir', type=str, default='data/LOL/eval15')
	parser.add_argument('--results_dir', type=str, default='results/seed42_e99')
	parser.add_argument('--out_dir', type=str, default='figures')
	config = parser.parse_args()
	main(config)
