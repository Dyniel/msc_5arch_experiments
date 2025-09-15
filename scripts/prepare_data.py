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

import os
import argparse
import zipfile
import logging
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_image_files(directory):
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    return [os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if os.path.splitext(file)[1].lower() in exts]

def process_image(image_path, resolution):
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        short, long = (h, w) if h < w else (w, h)
        crop_size = short
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((resolution, resolution), Image.LANCZOS)
        return img_resized
    except Exception as e:
        logging.warning(f"Could not process image {image_path}: {e}")
        return None

def create_nvlabs_zip(image_paths, output_path, resolution):
    logging.info(f"Creating NVLabs ZIP file at {output_path}...")
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing for ZIP")):
            img = process_image(image_path, resolution)
            if img:
                import io
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                zf.writestr(f'{i:08d}.png', buf.getvalue())
    logging.info("NVLabs ZIP file created successfully.")

def create_pggan_tfrecords(image_paths, output_path, resolution):
    logging.info(f"Creating TFRecords file at {output_path}...")
    with tf.io.TFRecordWriter(output_path) as writer:
        for image_path in tqdm(image_paths, desc="Processing for TFRecords"):
            img = process_image(image_path, resolution)
            if img:
                img_chw = np.array(img).transpose((2, 0, 1))
                feature = {
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img_chw.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_chw.tobytes()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    logging.info("TFRecords file created successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare image data for GAN training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source_dir", required=True, help="Path to the directory with raw images.")
    parser.add_argument("--output_dir", required=True, help="Path to save the processed ZIP and TFRecords files.")
    parser.add_argument("--resolution", type=int, default=256, help="Target resolution for images.")
    parser.add_argument("--name", default="dataset", help="Base name for the output files.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = find_image_files(args.source_dir)
    if not image_paths:
        logging.error(f"No images found in {args.source_dir}. Exiting.")
        return
    logging.info(f"Found {len(image_paths)} images.")
    zip_output_path = os.path.join(args.output_dir, f"{args.name}_{args.resolution}px.zip")
    create_nvlabs_zip(image_paths, zip_output_path, args.resolution)
    tfrecords_output_path = os.path.join(args.output_dir, f"{args.name}_{args.resolution}px.tfrecords")
    create_pggan_tfrecords(image_paths, tfrecords_output_path, args.resolution)
    logging.info("Data preparation complete.")

if __name__ == "__main__":
    main()
