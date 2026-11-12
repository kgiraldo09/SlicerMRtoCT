'''
Creative Commons Attribution-NonCommercial 4.0 International

Copyright (c) 2026 Kevin Estiven Giraldo Paniagua

This work is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License.
Full license text available at:
https://creativecommons.org/licenses/by-nc/4.0/legalcode

MR -> synthetic CT with the SAdaNGF_3d_unpaired generator. Standalone: no training repo needed.

Setup: put the trained weights next to this file

    mr_to_sct.py
    best_net_G_A.pth      <- from results/BrainMets_CHRU_all/SAdaNGF_3d_unpaired/

Usage:

    python mr_to_sct.py INPUT OUTPUT_FOLDER

    INPUT          an MR .nii.gz, or a folder: every .nii.gz in it (and its sub-folders) is translated
    OUTPUT_FOLDER  gets <name>_CT_SAdaNGF_3d_unpaired.nii.gz for each MR <name>.nii.gz (a trailing
                   _MR is dropped), in the same sub-folders as in INPUT: the sCT in HU, same grid
                   and affine as the MR

Steps for each MR (settings of the training config configs/BrainMets_CHRU_all/3d_inf.yaml):
    1. scale to [-1, 1] with percentiles 1 and 99.9, clipped
    2. resample to 1 x 1 x 1 mm (skipped when the MR is already at 1 mm)
    3. sliding-window inference: 128^3 patches, overlap 0.8, gaussian blending (as MONAI's inferer)
    4. resample back to the MR grid and convert to HU: [-1, 1] -> [-1000, 2000]

From other code (e.g. 3D Slicer):

    import mr_to_sct
    net = mr_to_sct.load_generator(mr_to_sct.WEIGHTS, device)
    sct_hu = mr_to_sct.mr_to_sct(mr_array_ijk, spacing_ijk, net, device)

Dependencies: numpy and torch only (+ nibabel and tqdm for the command line). No MONAI: the sliding
window is implemented here, so importing this file never loads torchvision or triton.
'''
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# Settings (from the test command and configs/BrainMets_CHRU_all/3d_inf.yaml)
# ---------------------------------------------------------------------------------------------
MODEL_NAME = 'SAdaNGF_3d_unpaired'
WEIGHTS = Path(__file__).resolve().parent / 'best_net_G_A.pth'   # next to this file

MR_PERCENTILES = (1.0, 99.9)        # ScaleIntensityRangePercentilesd(lower=1, upper=99.9, b_min=-1, b_max=1, clip=True)
CT_HU_RANGE = (-1000.0, 2000.0)     # ScaleIntensityRanged(a_min=-1000, a_max=2000, b_min=-1, b_max=1, clip=True)
TARGET_SPACING = 1.0                # mm, isotropic
SLIDING_WINDOW = dict(roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.8,
                      padding_mode='constant', cval=-1.0, mode='gaussian')


# ---------------------------------------------------------------------------------------------
# Generator: image branch of ResnetGenerator3 (netG resnet_9blocks3, 3D, ngf 64, instance norm,
# no dropout). Layer indices match the training code, so best_net_G_A.pth loads strictly.
# The alpha and sign branches are only used for training losses, not to produce the sCT.
# ---------------------------------------------------------------------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad3d(1), nn.Conv3d(ch, ch, 3), nn.InstanceNorm3d(ch), nn.ReLU(True),
            nn.ReflectionPad3d(1), nn.Conv3d(ch, ch, 3), nn.InstanceNorm3d(ch),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator3(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9):
        super().__init__()
        base = [nn.ReflectionPad3d(3), nn.Conv3d(input_nc, ngf, 7), nn.InstanceNorm3d(ngf), nn.ReLU(True)]
        for i in range(2):                                    # downsampling
            ch = ngf * 2 ** i
            base += [nn.Conv3d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm3d(ch * 2), nn.ReLU(True)]
        base += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]
        up = []
        for i in range(2):                                    # upsampling
            ch = ngf * 2 ** (2 - i)
            up += [nn.ConvTranspose3d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                   nn.InstanceNorm3d(ch // 2), nn.ReLU(True)]
        up += [nn.ReflectionPad3d(3), nn.Conv3d(ngf, output_nc, 7), nn.Tanh()]
        self.model_base = nn.Sequential(*base)
        self.model_upI = nn.Sequential(*up)

    def forward(self, x):
        return self.model_upI(self.model_base(x))


def load_generator(weights=WEIGHTS, device='cpu', ngf=64):
    """Builds the generator and loads best_net_G_A.pth. A missing or mismatching weight is an error."""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f'{weights} not found: copy best_net_G_A.pth '
                                f'(results/BrainMets_CHRU_all/{MODEL_NAME}/) next to mr_to_sct.py')
    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k.startswith(('model_base.', 'model_upI.'))}
    net = ResnetGenerator3(ngf=ngf)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


# ---------------------------------------------------------------------------------------------
# Pre / post-processing
# ---------------------------------------------------------------------------------------------
def normalize_mr(mr):
    """MR -> [-1, 1] with percentiles 1 and 99.9 of the whole volume, clipped."""
    mr = np.asarray(mr, dtype=np.float32)
    lo, hi = np.percentile(mr, MR_PERCENTILES)
    if hi <= lo:
        raise ValueError('The MR is constant: nothing to translate.')
    return np.clip((mr - lo) / (hi - lo) * 2 - 1, -1, 1).astype(np.float32)


def to_hu(sct):
    """Generator output in [-1, 1] -> Hounsfield units."""
    lo, hi = CT_HU_RANGE
    return ((np.clip(sct, -1, 1) + 1) / 2 * (hi - lo) + lo).astype(np.float32)


def resample(vol, scale, out_shape):
    """Trilinear resampling of a 3D tensor, keeping the corner of the field of view fixed.
    Output voxel j sits at input voxel coordinate (j + 0.5) * scale - 0.5 on each axis,
    i.e. scale = output spacing / input spacing."""
    axes = []
    for n_out, n_in, s in zip(out_shape, vol.shape, scale):
        j = torch.arange(n_out, dtype=torch.float32, device=vol.device)
        axes.append(2 * (j + 0.5) * s / n_in - 1)           # normalized coords, align_corners=False
    gi, gj, gk = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack((gk, gj, gi), dim=-1)[None]          # grid_sample wants (x, y, z) = (axis 2, 1, 0)
    out = F.grid_sample(vol[None, None].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out[0, 0]


def gaussian_weights(roi_size, device, sigma_scale=0.125):
    """Patch blending weights: separable gaussian (sigma = roi / 8), floored at 1e-3 (as MONAI)."""
    weights = None
    for n in roi_size:
        x = torch.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1, dtype=torch.float, device=device)
        x = torch.exp(x ** 2 / (-2 * (n * sigma_scale) ** 2))
        weights = x if weights is None else weights.unsqueeze(-1) * x
    return weights.clamp(min=max(weights.min().item(), 1e-3))


def patch_starts(size, roi, overlap):
    """Patch start positions along one axis (as MONAI's dense_patch_slices); the last patch ends at the edge."""
    step = roi if roi == size else max(int(roi * (1 - overlap)), 1)
    n = next((d for d in range(math.ceil(size / step)) if d * step + roi >= size), 0) + 1
    return [i * step - max(i * step + roi - size, 0) for i in range(n)]


def sliding_window(inputs, net, roi_size, sw_batch_size=1, overlap=0.25, cval=0.0,
                   mode='gaussian', padding_mode='constant'):
    """Gaussian-blended sliding-window inference on a (1, C, D, H, W) tensor, equivalent to MONAI's
    SlidingWindowInferer(mode='gaussian', padding_mode='constant') but with plain PyTorch only."""
    if mode != 'gaussian' or padding_mode != 'constant':
        raise ValueError('Only gaussian blending and constant padding are implemented.')
    spatial = list(inputs.shape[2:])
    before = [max(r - n, 0) // 2 for n, r in zip(spatial, roi_size)]          # pad volumes smaller than a patch
    after = [max(r - n, 0) - b for n, r, b in zip(spatial, roi_size, before)]
    pad = [p for b, a in zip(reversed(before), reversed(after)) for p in (b, a)]
    if any(pad):
        inputs = F.pad(inputs, pad, mode='constant', value=cval)
    size = list(inputs.shape[2:])

    windows = [tuple(slice(s, s + r) for s, r in zip(starts, roi_size))
               for starts in itertools.product(*(patch_starts(n, r, overlap) for n, r in zip(size, roi_size)))]
    weights = gaussian_weights(roi_size, inputs.device)
    out, count = None, torch.zeros(size, device=inputs.device)
    for b in range(0, len(windows), sw_batch_size):
        batch = windows[b:b + sw_batch_size]
        pred = net(torch.cat([inputs[(slice(None), slice(None), *w)] for w in batch]))
        if out is None:
            out = torch.zeros((1, pred.shape[1], *size), dtype=pred.dtype, device=inputs.device)
        for p, w in zip(pred, batch):
            out[(0, slice(None), *w)] += p * weights
            count[w] += weights
    out = out / count
    return out[(slice(None), slice(None), *(slice(b, b + n) for b, n in zip(before, spatial)))]


def mr_to_sct(mr_ijk, spacing_ijk, net, device, sliding_window_params=None):
    """
    mr_ijk      : MR as a numpy array in NIfTI voxel order (i, j, k), any intensity range
                  (nibabel: img.get_fdata(); 3D Slicer: arrayFromVolume(node).transpose(2, 1, 0))
    spacing_ijk : voxel size in mm along i, j, k (nibabel: img.header.get_zooms()[:3]; Slicer: node.GetSpacing())
    net         : generator from load_generator(), on `device`
    returns     : synthetic CT in HU, numpy float32, same shape and voxel grid as mr_ijk
    """
    spacing = [float(s) for s in spacing_ijk]
    x = torch.from_numpy(normalize_mr(mr_ijk))                                       # 1. [-1, 1]

    to_1mm = any(abs(s - TARGET_SPACING) > 1e-3 for s in spacing)                 # 2. 1 mm grid
    if to_1mm:
        shape_1mm = [max(1, round(n * s / TARGET_SPACING)) for n, s in zip(x.shape, spacing)]
        x = resample(x, [TARGET_SPACING / s for s in spacing], shape_1mm)

    sw = dict(SLIDING_WINDOW, **(sliding_window_params or {}))                      # 3. inference
    inputs = x[None, None].permute(0, 1, 4, 2, 3).to(device)                       # (1,1,k,i,j) as in training
    with torch.no_grad():
        out = sliding_window(inputs, net, **sw)
    sct = out[0, 0].permute(1, 2, 0).clamp(-1, 1).float().cpu()                   # back to (i, j, k)

    if to_1mm:                                                                     # 4. MR grid, HU
        sct = resample(sct, [s / TARGET_SPACING for s in spacing], mr_ijk.shape)
    return to_hu(sct.numpy())


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def find_cases(input_path, output_folder, pattern='*.nii.gz'):
    """[(MR path, sCT path)] for one MR file, or for every file matching `pattern` in a folder
    (sub-folders included). The sCT of a/b/X.nii.gz goes to OUTPUT_FOLDER/a/b/X_CT_<model>.nii.gz
    (a trailing _MR is dropped from X); sCTs already written by this script are never taken as input."""
    input_path, output_folder = Path(input_path).resolve(), Path(output_folder).resolve()
    suffix = f'_CT_{MODEL_NAME}.nii.gz'
    if input_path.is_dir():
        files = [f for f in sorted(input_path.rglob(pattern))
                 if f.is_file() and not f.name.endswith(suffix) and output_folder not in f.parents]
        rel_dirs = [f.parent.relative_to(input_path) for f in files]
    else:
        files, rel_dirs = [input_path], [Path('.')]
    cases = []
    for f, rel in zip(files, rel_dirs):
        name = f.name.removesuffix('.gz').removesuffix('.nii').removesuffix('_MR')
        cases.append((f, output_folder / rel / f'{name}{suffix}'))
    return cases


def main():
    import argparse

    import nibabel as nib
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=f'MR -> synthetic CT with {MODEL_NAME}.')
    parser.add_argument('input', type=Path, help='MR .nii.gz file, or folder: every .nii.gz in it is translated')
    parser.add_argument('output_folder', type=Path)
    parser.add_argument('--pattern', default='*.nii.gz', help='files to take in a folder (default: *.nii.gz)')
    parser.add_argument('--weights', type=Path, default=WEIGHTS, help='default: best_net_G_A.pth next to this file')
    parser.add_argument('--overwrite', action='store_true', help='redo cases that already have an sCT')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_generator(args.weights, device)
    cases = find_cases(args.input, args.output_folder, args.pattern)
    if not cases:
        raise SystemExit(f'No file in {args.input} matching {args.pattern}')
    print(f'{len(cases)} MR volume(s) -> {args.output_folder}  (weights {args.weights}, {device})')

    done = skipped = failed = 0
    for mr_path, out_path in tqdm(cases):
        name = mr_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            start = time.time()
            mr = nib.load(str(mr_path))
            sct = mr_to_sct(mr.get_fdata(dtype=np.float32), mr.header.get_zooms()[:3], net, device)
            out = nib.Nifti1Image(sct, mr.affine, mr.header)
            out.set_data_dtype(np.float32)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(out, str(out_path))
            done += 1
            tqdm.write(f'{name}: {time.time() - start:.1f} s')
        except Exception as e:  # keep going with the other cases
            failed += 1
            tqdm.write(f'{name} FAILED: {e}')
    print(f'Done: {done} written, {skipped} already there, {failed} failed.')


if __name__ == '__main__':
    main()
