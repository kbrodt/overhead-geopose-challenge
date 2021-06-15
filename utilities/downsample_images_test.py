import argparse
from pathlib import Path

import cv2
from misc_utils import load_image, save_image
from tqdm import tqdm


def downsample_images(args, downsample=2):
    indir = Path(args.indir)
    outdir = Path(args.outdir)

    outdir.mkdir(exist_ok=True)
    rgb_paths = list(indir.glob(f"*_RGB.{args.rgb_suffix}"))
    if rgb_paths == []:
        rgb_paths = list(indir.glob(f"*_RGB*.{args.rgb_suffix}"))  # original file names

    for rgb_path in tqdm(rgb_paths):
        # load
        rgb = load_image(
            rgb_path, args, use_cv=True
        )  # args.unit used to convert units on load
        assert len(rgb.shape) == 3

        # downsample
        if downsample > 1:
            target_shape = (
                int(rgb.shape[0] / downsample),
                int(rgb.shape[1] / downsample),
            )
            rgb = cv2.resize(rgb, target_shape)

        # save
        # units are NOT converted back here, so are in m
        #        save_image((outdir / rgb_path.name), rgb)
        save_image(
            (outdir / rgb_path.name.replace(args.rgb_suffix, "tif")), rgb
        )  # save as tif to be consistent with old code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, help="input directory", default=None)
    parser.add_argument("--outdir", type=str, help="output directory", default=None)
    parser.add_argument("--downsample", type=int, help="downsample image", default=1)
    parser.add_argument(
        "--nan-placeholder", type=int, help="placeholder value for nans", default=65535
    )
    parser.add_argument(
        "--unit", type=str, help="unit of AGLS (m, cm, or dm)", default="cm"
    )
    parser.add_argument(
        "--rgb-suffix",
        type=str,
        help="file extension for RGB data, e.g., tif or j2k",
        default="j2k",
    )
    args = parser.parse_args()
    print(args)
    downsample_images(args, args.downsample)
