# =============================================================================
# Copyright (c) Daniel Cieślak
#
# Thesis Title: Application of Deep Adversarial Networks for Synthesis
#               and Augmentation of Medical Images
#
# Author:       M.Sc. Eng. Daniel Cieślak
# Program:      Biomedical Engineering (WETI), M.Sc. full-time, 2023/2024
# Supervisor:   Prof. Jacek Rumiński, D.Sc., Eng.
#
# Notice:
# This script is part of the master's thesis and is legally protected.
# Copying, distribution, or modification without the author's permission is prohibited.
# =============================================================================

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse, json, time, logging, hashlib
from typing import List
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
import torch_fidelity

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
L = logging.getLogger("eval")

def list_images(root, exts=(".png",".jpg",".jpeg",".bmp",".webp")) -> List[str]:
    try: return [os.path.join(root,f) for f in os.listdir(root) if f.lower().endswith(exts)]
    except FileNotFoundError: return []

def precheck_images(folder, limit=0):
    files = list_images(folder); files = files[:limit] if limit>0 else files
    bad, ok = [], 0
    for i,p in enumerate(files,1):
        try:
            with Image.open(p) as im: im.verify()
            ok += 1
        except (UnidentifiedImageError, OSError):
            bad.append(p)
        if i%500==0: L.info(f"[precheck] {i}/{len(files)} checked, bad so far: {len(bad)}")
    return ok, bad

def hash_dir(path, max_files=50):
    h=hashlib.sha1(); files=sorted(list_images(path))[:max_files]
    for p in files:
        try:
            st=os.stat(p)
            h.update(p.encode()); h.update(str(st.st_size).encode()); h.update(str(int(st.st_mtime)).encode())
        except FileNotFoundError: pass
    return h.hexdigest()[:12]

def validate_metrics(d):
    for k,v in d.items():
        if k=="meta": continue
        if not (isinstance(v,float) and (np.isfinite(v) or np.isnan(v))):
            raise ValueError(f"Non-finite metric: {k}={v}")

def main(a):
    if not os.path.isdir(a.real): raise FileNotFoundError(a.real)
    if not os.path.isdir(a.fake): raise FileNotFoundError(a.fake)
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)

    device = torch.device("cpu")
    L.info(f"device={device.type} torch={torch.__version__}")
    L.info(f"real={a.real} fake={a.fake}")
    L.info(f"fingerprints real={hash_dir(a.real)} fake={hash_dir(a.fake)}")

    if a.precheck>0:
        L.info(f"[precheck] opening first {a.precheck} images in REAL and FAKE")
        ok_r,bad_r = precheck_images(a.real, a.precheck)
        ok_f,bad_f = precheck_images(a.fake, a.precheck)
        if bad_r or bad_f:
            L.error(f"Corrupt images detected. real_bad={len(bad_r)} fake_bad={len(bad_f)}")
            for p in (bad_r+bad_f)[:10]: L.error(f"bad: {p}")
            return
        L.info(f"[precheck] ok_real={ok_r} ok_fake={ok_f}")

    L.info("Calculating KID ...")
    t0=time.time()
    metrics = torch_fidelity.calculate_metrics(
        input1=a.real, input2=a.fake, cuda=False,
        isc=False, fid=False, kid=True, prc=False, verbose=True,
        kid_subset_size=a.kid_subset_size, kid_subsets=a.kid_subsets
    )
    L.info(f"KID done ({time.time()-t0:.1f}s)")

    results = {
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std":  float(metrics["kernel_inception_distance_std"]),
        "fid": float("nan"),
        "clean_fid": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "meta": {
            "device": device.type,
            "kid_subset_size": a.kid_subset_size,
            "kid_subsets": a.kid_subsets,
            "real_fp": hash_dir(a.real),
            "fake_fp": hash_dir(a.fake),
            "n_real": len(list_images(a.real)),
            "n_fake": len(list_images(a.fake)),
        }
    }

    L.info("--- Results ---")
    L.info(f'kid_mean: {results["kid_mean"]}')
    L.info(f'kid_std:  {results["kid_std"]}')

    validate_metrics(results)
    with open(a.out_json, "w") as f: json.dump(results, f, indent=2)
    L.info(f"saved: {a.out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GAN eval (CPU-only): KID; FID/PR disabled")
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--precheck", type=int, default=0)
    ap.add_argument("--kid-subset-size", type=int, default=256)
    ap.add_argument("--kid-subsets", type=int, default=50)
    args = ap.parse_args()
    main(args)
