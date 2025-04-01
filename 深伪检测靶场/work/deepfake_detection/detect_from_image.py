"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Xiaotian Si
"""
import os
import yaml
import argparse
from os.path import join
import cv2

import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np
from torch import package

# from dataset.transform import xception_default_data_transforms, transforms_380

from DeepfakeBench.training.detectors import DETECTOR
from minio import Minio
from minio.error import S3Error

from miniio_op import check_and_download, check_and_download_clip_basemodel

import glob

import sys

from facenet_pytorch.models.mtcnn import MTCNN
from torchvision import transforms as T

from adapt import T3A_v2

from log_utils import setup_logger
import csv

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)

logger = setup_logger()

BATCH_SIZE = 100
EXPAND_RATIO = 0.15

def get_detector():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN(
        margin=0,
        thresholds=[0.60, 0.60, 0.60],
        device=device,
        select_largest=True,  # the face with biggest size is ranked first
        keep_all=True)
    return face_detector

def adjust_bbox_ratio(bbox, avg_ratio, enlarge=False):
    xmin, ymin, xmax, ymax = bbox
    x_len = xmax - xmin
    y_len = ymax - ymin
    ratio = x_len / y_len
    if ratio == avg_ratio:
        return bbox
    elif ratio > avg_ratio:
        if enlarge:
            r_y_len = x_len / avg_ratio
            ymin -= (r_y_len - y_len) / 2
            ymax = ymin + r_y_len
        else:
            r_x_len = (y_len * avg_ratio)
            xmin += (x_len - r_x_len) / 2
            xmax = xmin + r_x_len
    else:
        if enlarge:
            r_x_len = y_len * avg_ratio
            xmin -= (r_x_len - x_len) / 2
            xmax = xmin + r_x_len
        else:
            r_y_len = (x_len / avg_ratio)
            ymin += (y_len - r_y_len) / 2
            ymax = ymin + r_y_len
    rect_ratio = (xmax - xmin) / (ymax - ymin)
    # if abs(rect_ratio - avg_ratio) > 1e-2:
    #     print(rect_ratio)
    assert abs(rect_ratio - avg_ratio) < 0.1
    return [xmin, ymin, xmax, ymax]

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

def to_tensor(img):
    """
    Convert an image to a PyTorch tensor.
    """
    return T.ToTensor()(img)

def normalize(img, config):
    """
    Normalize an image.
    """
    mean = config['mean']
    std = config['std']
    normalize = T.Normalize(mean=mean, std=std)
    return normalize(img)

def preprocess_image(image, config):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    size = config['resolution']
    images = []
    for img in image:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        image_trans = normalize(to_tensor(img), config).unsqueeze(0)
        images.append(image_trans)
    preprocessed_images = torch.concatenate(images)
    return preprocessed_images


# @torch.no_grad()
# def inference(model, data, config, device):
#     data = preprocess_image(data, config).to(device)
#     output = model({"image": data}, inference=True)['cls']
#     predictions = nn.Softmax(dim=1)(output)
#     c, prediction = torch.max(predictions, 1)    # argmax
#     prediction = prediction.cpu().numpy()
#     c = c.detach().cpu().numpy()
#     return prediction, output, c
@torch.no_grad()
def inference(model, data,paths, config, device): ##### 添加了路径信息
    data = preprocess_image(data, config).to(device)
    print(f"[****] dict paths: {paths}")
    output = model({"image": data,"path": paths}, inference=True)['cls'] ##### 添加了路径信息
    predictions = nn.Softmax(dim=1)(output)
    c, prediction = torch.max(predictions, 1)    # argmax
    prediction = prediction.cpu().numpy()
    c = c.detach().cpu().numpy()
    #print(f"[****] in raw detect: {prediction,c,output}")
    return prediction, output, c

def load_our_model(model_name, device):
    # parse options and load config
    # cur_path = os.getcwd()
    root_path = os.getcwd()
    # while True:
    #     if os.path.split(cur_path)[-1] == "deepmake-detection":
    #         root_path = os.path.split(cur_path)[0]
    #         break
    #     cur_path = os.path.split(cur_path)[0]


    detector_path = os.path.join(root_path, "DeepfakeBench/training/config/detector/%s.yaml" % model_name.lower()) 

    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    if config['model_name'] == 'clip':
        config['clip_path'] = os.path.join(project_dir, "clip", "clip-vit-base-patch16/home/ubuntu/openai/clip-vit-base-patch16/")

    model_class = DETECTOR[config['model_name']]
    
    model = model_class(config).to(device)

    weights_cache = os.path.join(root_path, "weights/%s.pth" % config['model_name'])

    check_and_download(config['model_name'], weights_cache)

    ckpt = torch.load(weights_cache, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    logger.info('===> 加载模型文件成功!')

    return model, config

def test_full_image_network(image_path, model_name, output_path, output_csv, cuda=True, adapt=False):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored

    :param cuda: enable cuda
    :return:
    """
    logger.info('开始检测文件: {}'.format(image_path))
    docker_root = "/df_detect/"

    # Read and write

    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(image_path) is False:
        imgs_paths = [image_path]
    else:
        imgs_paths = glob.glob(os.path.join(image_path, "*"))
    imgs = []

    for img_path in imgs_paths:
        image = pil_image.open(img_path).convert('RGB')

        image = np.array(image)
        imgs.append((image, img_path))

    # Face detector
    face_detector = get_detector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_our_model(model_name, device)

    model.eval()

    if args.adapt:
        model = T3A_v2(2,1,config,model)

    if model is None:
        logger.info("没有找到模型!")
        return

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1


    pbar = tqdm(total=len(imgs))
    prediction = None

    batch_frames = []
    batch_faces = []
    batch_paths = [] ### 多包含了路径信息
    batch_paths_final = [] ### 多包含了路径信息
    batch_frame_paths = [] ### 多包含了路径信息
    batch_cor = []
    video_fname = []

    cc = []
    csv_rows = []
    for batch_idx,(image,img_path) in enumerate(imgs): 

        video_fn = img_path.split('/')[-1]
        video_fname.append(video_fn)
        if image is None:
            break

        pbar.update(1)

        batch_frames.append(image)
        batch_paths.append(img_path) ### 多包含了路径信息


        if len(batch_frames) == BATCH_SIZE or (batch_idx == len(imgs)-1 and len(batch_frames) > 0):
            for i, batch_frame in enumerate(batch_frames):
                batch_frame = batch_frame[np.newaxis,:]
                # face detection
                boxes, probs, points = face_detector.detect(batch_frame,landmarks=True)

                frame = cv2.cvtColor(np.asarray(batch_frame[0]),cv2.COLOR_RGB2BGR)

                frame_boxes = boxes[0]
                frame_probs = probs[0]
                frame_points = points[0]
                if frame_boxes is None:
                    batch_cor.append(None)
                    continue  # no face is detected
                n_face = frame_boxes.shape[0]

                ori_boxes = frame_boxes[0]
                ori_points = frame_points[0]

                square_box = adjust_bbox_ratio(ori_boxes, 1., enlarge=True)
                square_box = [int(x) for x in square_box]
                xmin, ymin, xmax, ymax = square_box

                h, w, _ = frame.shape
                # ensure square
                x_w = xmax - xmin
                ymax = ymin + x_w
                pad = int(x_w * EXPAND_RATIO)
                ymin = max(ymin - pad, 0)
                ymax = min(ymax + pad, h)
                xmin = max(xmin - pad, 0)
                xmax = min(xmax + pad, w)
                x_w = xmax - xmin
                y_w = ymax - ymin
                if x_w < y_w:
                    ymin = ymin + (y_w - x_w)
                elif x_w > y_w:
                    xmin = xmin + (x_w - y_w)
                assert ymax - ymin == xmax - xmin

                cropped_face = frame[ymin:ymax, xmin:xmax]

                batch_faces.append(cropped_face)
                batch_paths_final.append(batch_paths[i]) #### 多包含了路径信息

                batch_cor.append((ymin,ymax,xmin,xmax))

            if len(batch_faces) == 0:
                batch_frames = []
                batch_faces = []
                batch_paths_final = [] ##### 多包含了路径信息
                batch_cor = []
                video_fname = []
                logger.info("no faces found")
                continue
            prediction, output, c = inference(model, batch_faces,batch_paths_final, config, device)  #### 多包含了路径信息
            
            gt_id = 0
            for idx, face in enumerate(batch_faces):
                if batch_cor[idx] != None:
                    label = 'fake' if prediction[gt_id] == 1 else 'real'
                    color = (0, 255, 0) if prediction[gt_id] == 0 else (0, 0, 255)
                    x = batch_cor[idx][2]
                    y = batch_cor[idx][0]
                    w = batch_cor[idx][3] - batch_cor[idx][2]
                    h = batch_cor[idx][1] - batch_cor[idx][0]
                    cv2.putText(batch_frames[idx], label + " " + "%.3f" % c[gt_id], (x, y+h+30),
                                font_face, font_scale,
                                color, thickness, 2)
                    if label == "fake":
                        cc.append(c[gt_id])
                        csv_rows.append([video_fname[idx], 1])
                    else:
                        cc.append(1-c[gt_id])
                        csv_rows.append([video_fname[idx], 0])
                    cv2.rectangle(batch_frames[idx], (x, y), (x + w, y + h), color, 2)
                    gt_id += 1       
                # Show
                image = pil_image.fromarray(batch_frames[idx])
                v_fname = video_fname[idx]
                video_fn1 = v_fname.split(".")

                if prediction[idx] is None:
                    video_fn = video_fn1[0]+ "." + video_fn1[1]
                elif prediction[idx]:
                    video_fn = video_fn1[0] + "_fake." + video_fn1[1]
                else:
                    video_fn = video_fn1[0] + "_real." + video_fn1[1]

                if prediction[idx] is not None:
                    logger.info("结果置信度：%s: %s" % (video_fn, np.mean(cc)))
                image.save(join(output_path, video_fn))

            batch_frames = []
            batch_faces = []
            batch_paths_final = []
            batch_cor = []
            video_fname = []

    pbar.close()
    # 保存csv
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in csv_rows:
            csv_writer.writerow(row)




if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--image_path', '-i', type=str)
    p.add_argument('--model_name', '-mi', type=str, default="Base")
    p.add_argument('--adapt', type=bool, default=False)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--output_csv', '-oc', type=str, default="./output_csv.csv")
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    if args.model_name.lower() == "clip_plus":
        args.model_name = "CLIP"
        args.adapt = True

    if args.model_name.lower() == "base":
        args.model_name = "xception"
    elif args.model_name.lower() == "selfblend":
        args.model_name = "efficientnetb4"
    if args.model_name.lower() == "clip":
        # check clip models
        os.makedirs(os.path.join(project_dir, "clip"),exist_ok=True)
        check_and_download_clip_basemodel(os.path.join(project_dir, "clip", "clip-vit-base-patch16"))

    test_full_image_network(**vars(args))

