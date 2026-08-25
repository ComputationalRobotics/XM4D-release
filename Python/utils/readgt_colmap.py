import pandas as pd
import numpy as np

from utils.io import load_matrix_from_bin, save_matrix_to_bin

import re
from pathlib import Path

def frame_id_from_name(name):
    """
    Extract frame id only when the filename stem ends with an integer token.

    Accepted:
        frame000091.jpg -> 91
        000091.png -> 91
        image_12.jpg -> 12

    Rejected:
        111.111.jpg
        frame001.25.jpg
        frameabc.jpg
    """
    stem = Path(name).stem

    # Reject decimal-like names explicitly.
    if re.search(r"\d+\.\d+", stem):
        raise ValueError(f"Decimal-like filename; refusing to extract frame id: {name}")

    # Accept trailing integer only.
    match = re.search(r"(?<!\d)(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not extract full-integer frame id from image name: {name}")

    return int(match.group(1))

# The reconstructed pose of an image is specified as the projection from world to the camera coordinate
def quat2rot(qw, qx, qy, qz):
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R

def load_camera_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue  
        parts = line.strip().split()

        camera_id = int(parts[0])  # CAMERA_ID
        model = parts[1]  # MODEL
        width = int(parts[2])  # WIDTH
        height = int(parts[3])  # HEIGHT
        params = list(map(float, parts[4:]))  # PARAMS start from fifth
        K = np.eye(3)
        if model == "SIMPLE_PINHOLE":
            # params = [f, cx, cy]
            K[0, 0] = params[0]  # f
            K[1, 1] = params[0]  # f
            K[0, 2] = params[1]  # cx
            K[1, 2] = params[2]  # cy
        elif model == "PINHOLE":
            # params = [fx, fy, cx, cy]
            K[0, 0] = params[0]  # fx
            K[1, 1] = params[1]  # fy
            K[0, 2] = params[2]  # cx
            K[1, 2] = params[3]  # cy
        elif model in {"SIMPLE_RADIAL", "RADIAL"}:
            # Distortion is ignored here; the static solver only needs K.
            K[0, 0] = params[0]
            K[1, 1] = params[0]
            K[0, 2] = params[1]
            K[1, 2] = params[2]
        elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            K[0, 0] = params[0]
            K[1, 1] = params[1]
            K[0, 2] = params[2]
            K[1, 2] = params[3]
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        data.append((camera_id, K, width, height))
    
    camera_df = pd.DataFrame(data, columns=['CAMERA_ID', 'K', 'width', 'height'])
    camera_df.set_index('CAMERA_ID', inplace=True) 
    
    return camera_df

def load_colmap_camera(gt_path):
    camera_path = gt_path + "/sparse/cameras.txt"
    camera_df = load_camera_data(camera_path)
    results = {}
    for camera_id, row in camera_df.iterrows():
        K = row['K']
        results[camera_id] = {
            "model" : "PINHOLE",
            "width" :row['width'],
            "height": row['height'],
            "params": [K[0,0], K[1,1], K[0,2], K[1,2]]
        }
    return results

def _is_image_metadata_line(parts):
    """A COLMAP images.txt metadata row is
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    i.e. >=10 tokens whose last token (NAME) is a filename, not a number.
    The alternating POINTS2D rows are empty or all-numeric, so this
    structural test parses every image regardless of whether POINTS2D
    lines are present, avoiding the every-other-frame desync that a
    positional toggle hits on empty POINTS2D lines.
    """
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        int(parts[8])
    except ValueError:
        return False
    try:
        float(parts[9])  # numeric -> a POINTS2D token, not an image name
        return False
    except ValueError:
        return True


def load_image_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    image_data = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#") or not line.strip():
            continue
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if not _is_image_metadata_line(parts):
            continue
        image_id = int(parts[0])  # IMAGE_ID
        qw, qx, qy, qz = map(float, parts[1:5])  # quat
        tx, ty, tz = map(float, parts[5:8])  # trans
        camera_id = int(parts[8])  # CAMERA_ID
        name = " ".join(parts[9:])  # IMAGE NAME

        pose = {"qw": qw, "qx": qx, "qy": qy, "qz": qz, "tx": tx, "ty": ty, "tz": tz}
        image_data.append((name, image_id, camera_id, pose))

    df = pd.DataFrame(image_data, columns=['NAME', 'IMAGE_ID', 'CAMERA_ID', 'POSE'])
    df.set_index('NAME', inplace=True)
    return df

def load_gt_depth(gt_path):
    image_path = gt_path + "/images.txt"
    image_df = load_image_data(image_path)
    gt_depth, _ = load_matrix_from_bin(gt_path + "/depth_gt.bin")

    gt_depth = gt_depth[:,(0,1,2,4)]

    image_id_to_name = image_df.reset_index().set_index('IMAGE_ID')['NAME']

    gt_depth_df = pd.DataFrame(gt_depth, columns=['IMAGE_ID', 'COORD1', 'COORD2', 'DEPTH'])
    gt_depth_df['NAME'] = gt_depth_df['IMAGE_ID'].map(image_id_to_name)  # 添加 NAME 列

    grouped = gt_depth_df.groupby('NAME').apply(
        lambda x: {
            'COORD1': x['COORD1'].to_numpy(),
            'COORD2': x['COORD2'].to_numpy(),
            'DEPTH': x['DEPTH'].to_numpy()
        }
    )
    return grouped
    

def load_colmap_gt(gt_path):
    camera_path = gt_path + "/sparse/cameras.txt"
    camera_df = load_camera_data(camera_path)

    image_path = gt_path + "/sparse/images.txt"
    image_df = load_image_data(image_path)

    results = {}
    for name, row in image_df.iterrows():
        pose = row["POSE"]
        camera_id = row["CAMERA_ID"]

        R = quat2rot(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
        t = np.array([pose["tx"], pose["ty"], pose["tz"]])

        if camera_id in camera_df.index:
            K = camera_df.loc[camera_id, "K"]
        else:
            K = None

        frame_id = frame_id_from_name(name)

        results[name] = {
            "id": frame_id,          # now from image name, e.g. frame000091.jpg -> 91
            "K": K,
            "R": R,
            "t": t,
            "camera_id": camera_id,  # keep original COLMAP camera id separately
        }

    return results
