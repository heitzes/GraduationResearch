# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os, glob
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    #range_re = re.compile(r'^(\d+)-(\d+)$')
    #m = range_re.match(s)
    #if m:
    #    return list(range(int(m.group(1)), int(m.group(2))+1))
    if "," in s:
        vals = s.split(',')
        return [int(x) for x in vals]
    elif "-" in s:
        vals = s.split('-')
        return list(range(int(vals[0]), int(vals[1]) + 1))
    elif "/" in s:
        vals = s.split('/')

        return vals
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
# Get targets' name from --target
@click.option('--target', 'target_fname', type=num_range, help='Target image file to project to', required=True, metavar='FILE')
@click.option('--origin', 'origin_fname', type=str, help='Target image file to project to', required=True, metavar='FILE')

def generate_style_mix(
    network_pkl: str,
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    target_fname: List[str],
    origin_fname: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    os.makedirs(outdir, exist_ok=True)

    def VGG_sample(name):
        vgg_path = os.path.join(outdir, name)
        imgs = sorted(glob.glob(os.path.join(vgg_path, '*.*g')))
        print(imgs)
        npz_path = os.path.join(vgg_path, "projected_w")
        npzs = sorted(glob.glob(os.path.join(npz_path, '*.*z')))
        print(npzs)
        return imgs, npzs

    img_path, npz_path = VGG_sample(origin_fname)
    row_seeds = []
    rs = []
    w_dict = {}
    image_dict = {}
    for npz in npz_path:
        print(npz[-8:-4])
        ws = np.load(npz)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        rs.append(ws[0])
    for i in range(len(img_path)):
        print(img_path[i][-8:-4], npz_path[i][-8:-4])
        row_seeds.append(img_path[i][-8:-4])
        w_dict[row_seeds[i]] = rs[i]
    for i in range(len(img_path)):
        img = img_path[i]
        target_pil = PIL.Image.open(img).convert('RGB')
        # w, h = target_pil.size
        # s = min(w, h)
        # target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        image_dict[(row_seeds[i], row_seeds[i])] = target_uint8


    # col_seeds가 0일때
    path = os.path.dirname(os.path.abspath(__file__))
    print(path)
    target_path = []
    prow_path = []
    col_seeds = []
    for t in target_fname:
        target_path.append(os.path.join(path, t+'.png'))
        prow_path.append(os.path.join(path, 'out_dir', 'projected_'+t+'.npz'))
    wss = []
    for wp in prow_path:
        ws = np.load(wp)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        wss.append(ws[0])
    for i in range(len(target_fname)):
        col_seeds.append(i)
        w_dict[i] = wss[i]
    for t in range(len(target_path)):
        tp = target_path[t]
        target_pil = PIL.Image.open(tp).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        image_dict[(t, t)] = target_uint8

    # if projected_w is not None:
    #     ws = np.load(projected_w)['w']
    #     ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
    #     assert ws.shape[1:] == (G.num_ws, G.w_dim)
    #     #print("ws[0].shape: ", ws[0].shape)
    #     #print("ws.shape: ", ws.shape)
    #
    #     # 차은우 w 콜럼에 어펜드
    #     #print(w_dict[100].shape)
    #     w_dict[-1] = ws[0]
    #     w_dict[-2] = ws[0]
    #
    #     # Load target image.
    #     target_pil = PIL.Image.open(target_fname).convert('RGB')
    #     w, h = target_pil.size
    #     s = min(w, h)
    #     target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    #     target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    #     target_uint8 = np.array(target_pil, dtype=np.uint8)
    #
    #     # 차은우 원본 저
    #     image_dict[(-1, -1)] = target_uint8
    #     image_dict[(-2, -2)] = target_uint8
    #     col_seeds.append(-1)
    #     row_seeds.append(-2)

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

    # print('Saving images...')
    # os.makedirs(outdir, exist_ok=True)
    # for (row_seed, col_seed), image in image_dict.items():
    #     PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')
    print(row_seeds, col_seeds)
    print('Saving image grid...')
    save_dir = os.path.join('./', outdir, origin_fname)
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save('{}/projected_w/grid_{}to{}.png'.format(save_dir, col_styles[0], col_styles[-1]))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------