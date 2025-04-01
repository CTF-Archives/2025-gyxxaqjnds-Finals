import torch
from PIL import Image 
import os, json, glob
import cv2
import pandas as pd
import numpy as np
import dlib
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='process face image from video.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-data_path', type=str, help=' video data directory.')

    args = parser.parse_args()

    return args

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def main(args):
    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    video_path = args.data_path

    videos = glob.glob(os.path.join(video_path, "*.mp4"))

    pbar = tqdm(total=len(videos))
    for video in videos:
        video_root_path = video #os.path.join(video_path, video)
        image_path = video_root_path.split(".mp4")[0]
        if os.path.exists(image_path) is False:
            os.mkdir(image_path)
        reader = cv2.VideoCapture(video)
        # fps = reader.get(cv2.CAP_PROP_FPS)
        # num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0

        while reader.isOpened():
            _, image = reader.read()
            if image is None:
                break
            frame_num += 1
            # Image size
            height, width = image.shape[:2]
            # 2. Detect with dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 1)
            if len(faces):
                # For now only take biggest face
                face = faces[0]
                # --- Prediction ---------------------------------------------------
                # Face crop with dlib and bounding box scale enlargement
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                cv2.imwrite(image_path + "/" + str(frame_num) + ".png", cropped_face)
        pbar.update(1)


args = parse_args()
main(args)