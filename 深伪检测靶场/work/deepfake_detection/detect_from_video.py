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

import imageio.v2 as iio

# from network.models import model_selection
# from dataset.transform import xception_default_data_transforms, transforms_380
from torch import package
import sys

from DeepfakeBench.training.detectors import DETECTOR
from minio import Minio
from minio.error import S3Error

from miniio_op import check_and_download, check_and_download_clip_basemodel

from facenet_pytorch.models.mtcnn import MTCNN
from torchvision import transforms as T

from adapt import T3A_v2

from log_utils import setup_logger
import csv

project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)

logger = setup_logger()

BATCH_SIZE = 32
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


@torch.no_grad()
def inference(model, data, config, device):
    data = preprocess_image(data, config).to(device)
    output = model({"image": data}, inference=True)['cls']
    predictions = nn.Softmax(dim=1)(output)
    c, prediction = torch.max(predictions, 1)    # argmax
    prediction = prediction.cpu().numpy()
    c = c.detach().cpu().numpy()
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
    logger.info('===> 加载模型成功!')

    return model, config


def test_full_image_network(video_path, model_name, output_path,output_csv,
                            start_frame=0, end_frame=None, cuda=True, adapt=False):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    logger.info('开始检测文件: {}'.format(video_path))
    docker_root = "/df_detect/"
    # docker_root = "/home/sxt/deepmake-detection"

    # Read and write
    reader = cv2.VideoCapture(video_path)

    video_fn = video_path.split('/')[-1].split('.')[0]+'.mp4'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(num_frames)
    # pause()
    writer = None

    # Face detector
    face_detector = get_detector()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_our_model(model_name, device)

    model.eval()

    if args.adapt:
        model = T3A_v2(2,1,config,model)

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)
    total=end_frame-start_frame
    fake_num = 0
    fake_c = []

    batch_frames = []
    batch_faces = []
    batch_cor = []

    # Init output writer
    if writer is None:
        writer = iio.get_writer(join(output_path, video_fn), format='ffmpeg', mode='I', fps=fps, codec='libx264', pixelformat='yuv420p')

    while frame_num < num_frames:
        _, image = reader.read()
        # print(image.shape)
        # pause()
        if image is None:
            break
        frame_num += 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = pil_image.fromarray(image)
        image = np.array(image)
        batch_frames.append(image)

        pbar.update(1)

        if len(batch_frames) == BATCH_SIZE or (frame_num == num_frames - 1
                                               and len(batch_frames) > 0):

            for i, batch_frame in enumerate(batch_frames):
                batch_frame = batch_frame[np.newaxis,:]
                # face detection
                boxes, probs, points = face_detector.detect(batch_frame,
                                                            landmarks=True)

                frame = cv2.cvtColor(np.asarray(batch_frame[0]),
                                    cv2.COLOR_RGB2BGR)
                
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
                batch_cor.append((ymin,ymax,xmin,xmax))

            if len(batch_faces) == 0:
                batch_frames = []
                batch_faces = []
                batch_cor = []
                continue

            prediction, output, c = inference(model, batch_faces, config, device)

            gt_id = 0
            for idx, face in enumerate(batch_frames):

                if batch_cor[idx] != None:
                    label = 'fake' if prediction[gt_id] == 1 else 'real'
                    if label == "fake":
                        fake_num += 1
                    color = (0, 255, 0) if prediction[gt_id] == 0 else (0, 0, 255)
                    x = batch_cor[idx][2]
                    y = batch_cor[idx][0]
                    w = batch_cor[idx][3] - batch_cor[idx][2]
                    h = batch_cor[idx][1] - batch_cor[idx][0]
                    cv2.putText(batch_frames[idx], label + " " + "%.3f" % c[gt_id], (x, y+h+30),
                                font_face, font_scale,
                                color, thickness, 2)
                    if label == "fake":
                        fake_c.append(c[gt_id])
                    else:
                        fake_c.append(1-c[gt_id])
                    cv2.rectangle(batch_frames[idx], (x, y), (x + w, y + h), color, 2)
                    gt_id += 1
                # image = pil_image.fromarray(batch_frames[idx]).convert('RGB')
                writer.append_data(batch_frames[idx])


            batch_frames = []
            batch_faces = []
            batch_cor = []

    pbar.close()
    if writer is not None:
        # writer.release()
        writer.close()
        logger.info('完成! 输出存储在 {}'.format(join(output_path, video_fn)))
    else:
        logger.info('输入视频为空')
    csv_rows = []
    if len(fake_c) == 0:
        logger.info("no faces found in %s" % video_fn1)
        os.system("mv %s %s" % (join(output_path, video_fn), join(output_path, video_fn1)))
    elif fake_num >= int((end_frame-start_frame)/2):
        video_fn1 = video_fn.split(".mp4")[0]
        video_fn1 = video_fn1 + "_fake" + ".mp4"
        logger.info("置信度 %s: %s" % (video_fn1, np.mean(fake_c)))
        os.system("mv %s %s" % (join(output_path, video_fn), join(output_path, video_fn1)))
        csv_rows.append([video_fn,1])
    else:
        video_fn1 = video_fn.split(".mp4")[0]
        video_fn1 = video_fn1 + "_real" + ".mp4"
        logger.info("置信度 %s: %s" % (video_fn1, np.mean(fake_c)))
        os.system("mv %s %s" % (join(output_path, video_fn), join(output_path, video_fn1)))
        csv_rows.append([video_fn,0])
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in csv_rows:
            csv_writer.writerow(row)



if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
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

    os.makedirs(args.output_path, exist_ok=True)
    csv_path = args.output_csv
    if os.path.exists(csv_path) and os.path.isfile(csv_path):
        # 如果路径存在且为文件，则删除文件
        os.remove(csv_path)
    
    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))
