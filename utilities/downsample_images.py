import argparse
import multiprocessing
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from misc_utils import load_image, load_vflow, save_image


def load_save(rgb_path):
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    
    # load
    agl_path = rgb_path.with_name(
        rgb_path.name.replace("_RGB", "_AGL")
    ).with_suffix(".tif")
    vflow_path = rgb_path.with_name(
        rgb_path.name.replace("_RGB", "_VFLOW")
    ).with_suffix(".json")
    rgb = load_image(rgb_path, args, use_cv=True)  # args.unit used to convert units on load
    assert len(rgb.shape) == 3
    agl = load_image(agl_path, args)  # args.unit used to convert units on load
    _, _, _, vflow_data = load_vflow(
        vflow_path, agl, args
    )  # arg.unit used to convert units on load

    # downsample
    if args.downsample > 1:
        target_shape = (int(rgb.shape[0] / args.downsample), int(rgb.shape[1] / args.downsample))
        rgb = cv2.resize(rgb, target_shape)
        agl = cv2.resize(agl, target_shape, interpolation=cv2.INTER_NEAREST)
        vflow_data["scale"] /= args.downsample

    # save
    # units are NOT converted back here, so are in m
#    save_image((outdir / rgb_path.name), rgb)
    save_image((outdir / rgb_path.name.replace(args.rgb_suffix, "tif")), rgb) # save as tif to be consistent with old code
    save_image((outdir / agl_path.name), agl)
    with open((outdir / vflow_path.name), "w") as outfile:
        json.dump(vflow_data, outfile)


def downsample_images(args):
    indir = Path(args.indir)
    outdir = Path(args.outdir)

    outdir.mkdir(exist_ok=True)
    rgb_paths = list(indir.glob(f"*_RGB.{args.rgb_suffix}"))
    if rgb_paths == []:
        rgb_paths = list(indir.glob(f"*_RGB*.{args.rgb_suffix}")) # original file names

    # for rgb_path in tqdm(rgb_paths):
    #     load_save(rgb_path, args)
    
    with multiprocessing.Pool(args.n_jobs) as p:
        _ = list(p.imap_unordered(func=load_save, iterable=tqdm(rgb_paths)))


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
    parser.add_argument("--n-jobs", type=int, help="downsample image", default=20)
    args = parser.parse_args()
    print(args)
    downsample_images(args)
