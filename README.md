# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

> **CISC 867 course project fork.** This repository starts from the original Zero-DCE code and adds a reproduction of Zero-DCE on the LOL dataset, run on an Apple M2 Pro GPU. Our changes and additions are described in [Reproduction on LOL (CISC 867)](#reproduction-on-lol-cisc-867) at the end of this README. Everything above that section is the authors' original README.

You can find more details here: https://li-chongyi.github.io/Proj_Zero-DCE.html. Have fun!

**The implementation of Zero-DCE is for non-commercial use only.**

We also provide a MindSpore version of our code: https://pan.baidu.com/s/1uyLBEBdbb1X4QVe2waog_g (passwords: of5l).

# Pytorch 
Pytorch implementation of Zero-DCE

## Requirements
1. Python 3.7 
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0

Zero-DCE does not need special configurations. Just basic environment. 

Or you can create a conda environment to run our code like this:
conda create --name zerodce_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch

### Folder structure
Download the Zero-DCE_code first.
The following shows the basic folder structure.
```

├── data
│   ├── test_data # testing data. You can make a new folder for your testing data, like LIME, MEF, and NPE.
│   │   ├── LIME 
│   │   └── MEF
│   │   └── NPE
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # Zero-DEC network
├── dataloader.py
├── snapshots
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```
### Test: 

cd Zero-DCE_code
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
1) cd Zero-DCE_code

2) download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA">baidu cloud [password: 1234]</a>

3) unzip and put the  downloaded "train_data" folder to "data" folder
```
python lowlight_train.py 
```
##  License
The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License.


## Bibtex

```
@inproceedings{Zero-DCE,
 author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
 title = {Zero-reference deep curve estimation for low-light image enhancement},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {1780-1789},
 month = {June},
 year = {2020}
}
```

(Full paper: http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

## Contact
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com or Chunle Guo at guochunle@tju.edu.cn.

## TensorFlow Version 
Thanks tuvovan (vovantu.hust@gmail.com) who re-produces our code by TF. The results of TF version look similar with our Pytorch version. But I do not have enough time to check the details.
https://github.com/tuvovan/Zero_DCE_TF


---

## Reproduction on LOL (CISC 867)

This section describes what we changed and added to the original repository for the CISC 867 midterm report (*P02 Zero-Reference Low-Light Image Enhancement*). It contains what is needed to reproduce the report's Table 2, Figure 1 and Discussion.

### Results (LOL-v1 test set, 15 images, epoch 99)

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Published baseline (Guo et al., 2020; SICE Part 2 test set) | 16.57 | 0.59 |
| Authors' released model (`snapshots/Epoch99.pth`), evaluated by us on LOL | 14.86 | 0.562 |
| **Our reproduction** (mean ± std over 3 runs) | **16.99 ± 0.19** | **0.559 ± 0.005** |

Individual runs: 16.86 / 0.562 (unseeded), 16.89 / 0.561 (seed 42), 17.21 / 0.554 (seed 123).

All results use the epoch-99 checkpoint (training runs for 200 epochs, numbered 0–199) to match the authors' released model, `Epoch99.pth`, which was trained for the same number of epochs.

### Changes to the original files

- **`lowlight_train.py`, `Myloss.py`**: `.cuda()` replaced by automatic device selection (CUDA if available, else Apple MPS, else CPU).
- **`dataloader.py`**: `Image.ANTIALIAS` replaced by `Image.LANCZOS` (the same filter; the old name was removed in Pillow 10).
- **`lowlight_train.py`**: new optional arguments. Without them, training behaves exactly like the original code.
  - `--seed N` seeds weight initialisation and the data order.
  - `--paper_losses`, `--paper_optim` and `--zero_bias` switch to the loss equations, loss weights, optimiser settings and bias initialisation described in the paper (used only for the "matching the paper" experiment below).
- **`Myloss.py`**: new loss classes (`L_spa_paper`, `L_exp_paper`, `L_color_paper`, `L_TV_paper`) that follow the paper's equations. The original classes are unchanged.

`model.py` and `lowlight_test.py` are unchanged. The training settings used for our results are the original defaults: Adam, learning rate 1e-4, weight decay 1e-4, gradient clipping 0.1, batch size 8, 200 epochs, images resized to 256×256.

### Added files (in `Zero-DCE_code/`)

```
evaluate_lol.py            # evaluates a model on LOL: PSNR and SSIM
make_figure1.py            # creates Figure 1 of the report
snapshots_run1_unseeded/   # our retrained models at epoch 99
snapshots_seed42/
snapshots_seed123/
snapshots_paper_seed42/    # model from the "matching the paper" experiment
results/                   # summary.json + per_image.csv for every evaluated model
  seed42_e99/              #   (+ the enhanced test images used in Figure 1)
logs/                      # full training logs
figures/                   # Figure 1 (PDF and PNG)
```

`requirements.txt` (repository root) lists the exact package versions we used.

### Setup and data

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The datasets are not included. Download them into `Zero-DCE_code/data/`:

1. **Training data (2,002 images):** the `train_data` folder from the Google Drive link in the Train section above, unzipped to `Zero-DCE_code/data/train_data/`.
2. **LOL-v1 test set:** from the official LOL link (Wei et al., 2018) or the mirror [geekyrakshit/LoL-Dataset](https://huggingface.co/datasets/geekyrakshit/LoL-Dataset). Place `eval15` at `Zero-DCE_code/data/LOL/eval15/`, with `low/` and `high/` subfolders of 15 images each.

### Reproducing the results

Run all commands from `Zero-DCE_code/`.

Evaluate the provided models (no training needed):

```bash
python evaluate_lol.py --weights snapshots/Epoch99.pth                 --out_dir results/authors_pretrained
python evaluate_lol.py --weights snapshots_run1_unseeded/Epoch99.pth   --out_dir results/run1_unseeded_e99
python evaluate_lol.py --weights snapshots_seed42/Epoch99.pth          --out_dir results/seed42_e99
python evaluate_lol.py --weights snapshots_seed123/Epoch99.pth         --out_dir results/seed123_e99
```

Retrain (about 3¼ hours per run on an M2 Pro):

```bash
python lowlight_train.py --snapshots_folder snapshots_run1_unseeded/
python lowlight_train.py --seed 42  --snapshots_folder snapshots_seed42/
python lowlight_train.py --seed 123 --snapshots_folder snapshots_seed123/
```

The default `--snapshots_folder` is `snapshots/`, so training without that argument overwrites the authors' `snapshots/Epoch99.pth` at epoch 99. Apple's MPS backend is not fully deterministic, so retrained numbers can differ slightly from ours.

Recreate Figure 1 (after evaluating the seed-42 model):

```bash
python make_figure1.py
```

### "Matching the paper" experiment

The report notes that updating the code to match the paper made the results worse. We reran training (seed 42) with the paper's loss equations and weights (1, 1, 0.5, 20), plain Adam and zero bias initialisation:

```bash
python lowlight_train.py --seed 42 --paper_losses --paper_optim --zero_bias --snapshots_folder snapshots_paper_seed42/
```

At epoch 99 it scores 12.02 dB PSNR and 0.430 SSIM, and the outputs are much brighter and noisier than with the released code (`results/paper_seed42_e99/`).

### Attribution

Zero-DCE, its code (`model.py`, `Myloss.py`, `dataloader.py`, `lowlight_train.py`, `lowlight_test.py`) and its pretrained model are by Guo et al. and released for non-commercial academic use (see the License section above). The evaluation and figure scripts, the code changes above and our trained models were created for this course project.
