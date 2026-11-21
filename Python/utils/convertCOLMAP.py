

import numpy as np


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(HEADER)
        for camera_id in sorted(cameras.keys()):
            camera = cameras[camera_id]
            to_write = [camera_id, camera['model'], camera['width'], camera['height']] + camera['params']
            line = " ".join([str(elem) for elem in to_write])
            f.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    mean_observations = 0
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)
 
    with open(path, "w", encoding='utf-8') as fid:
        fid.write(HEADER)
        for image_id in sorted(images.keys()):
            image = images[image_id]
            image_header = [
                image['id'],
                *image['qvec'], 
                *image['tvec'], 
                image['camera_id'], 
                image['name']
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")
 
            points_strings = []
            fid.write(" ".join(points_strings) + "\n")

def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    mean_track_length = 0
    HEADER = "# 3D point list with one line of data per point:\n" + \
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" + \
             "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
 
    with open(path, "w", encoding='utf-8') as fid:
        fid.write(HEADER)
        for pt_id in sorted(points3D.keys()):
            pt = points3D[pt_id]
            point_header = [pt["id"], *pt["xyz"], *pt["rgb"], pt["error"]]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            fid.write(" ".join(track_strings) + "\n")


def _rotation_matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec.astype(np.float64)
