import os
import csv
import cv2
import shutil
import sys
import glob
import json
import math
import time
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import nn
from PIL import Image
from numpy import mean
from torch import optim
from torch import package
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.net import ResNetModel
from utils.parser import ConfigParser
from utils.loader import dataLoader
from tqdm import tqdm
from utils import FormatConverter
from model import Net, Net1
import sys
import traceback
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler

# 创建日志文件夹
log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志文件路径和名称
log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_adversarial-defense.log")

# 配置日志
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1, backupCount=7, encoding='utf-8')
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger('AdversarialDefense')
logger.setLevel(logging.INFO)  # 默认日志级别为INFO
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """全局捕获程序中断并记录到日志"""
    if issubclass(exc_type, KeyboardInterrupt):
        # 忽略Ctrl+C中断
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("发生异常程序中断", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = log_uncaught_exceptions


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """自定义错误处理，将错误信息记录到日志中"""
        logger.error("命令行参数错误: %s", message)
        self.print_help()  # 打印帮助信息
        sys.exit(2)  # 退出程序，状态码为2


def parse_args():
    parser = CustomArgumentParser(
        description='provides adversarial attacks to generate adversarial examples.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model_path', type=str, help='model directory.')
    parser.add_argument('-d', '--dataset', type=str, help='data for evaluation.')
    parser.add_argument('--model_arch', type=str, default='res50',
                        choices=['res18', 'res34', 'res50'],
                        help='Model architecture to use, like resnet18 or resnet34 or resnet50.')
    parser.add_argument('--algos', dest="algos", nargs="+",
                        help='set defense algorithm.')
    parser.add_argument('-outputratename', type=str, default="rate", help='save result.')
    parser.add_argument('-outputnumname', type=str, default="num", help='save result.')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小，默认为8')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader使用的工作进程数量，默认为0')
    parser.add_argument('--output', type=str, help='save result.')
    parser.add_argument('--csv_output', type=str, help='Path to save the CSV result.')

    args = parser.parse_args()

    # 检查必需的参数是否缺失
    if not args.model_path:
        logger.error("缺少模型路径参数 '--model_path'. 请指定模型路径。")
        sys.exit(2)  # 如果缺少模型路径参数，则退出程序，错误码为2

    if not args.dataset:
        logger.error("缺少数据集路径参数 '-d'. 请指定数据集路径。")
        sys.exit(2)  # 如果缺少数据集路径参数，则退出程序，错误码为2

    if not args.output:
        logger.error("缺少输出目录参数 '--output'. 请指定输出目录。")
        sys.exit(2)  # 如果缺少输出目录参数，则退出程序，错误码为2

    # 检查防御算法参数是否缺失
    if not args.algos:
        logger.error("缺少防御方法参数 '--algos'. 请指定一个防御方法。")
        sys.exit(2)  # 如果缺少防御方法参数，则退出程序，错误码为2

    logger.info("所有命令行参数解析成功!")
    return args  # 返回解析后的命令行参数


ifRGB = True


class BaseDataset(Dataset):
    def __init__(self, root, transform=None):
        super(BaseDataset, self).__init__()
        self.root = root
        self.transform = transform
        assert transform is not None, "transform is None"
        self.imgs = glob.glob(os.path.join(root, "*"))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        if ifRGB:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return (image, img_path)

    def __len__(self):
        return len(self.imgs)


class General_Data(BaseDataset):
    def __init__(self, root, transform=None):
        super(General_Data, self).__init__(root=root, transform=transform)
        imgs = glob.glob(os.path.join(root, "*"))
        imgs = [(i,) for i in imgs]
        self.imgs = imgs


def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                                            batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                                            batch_size].to(device)
            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
    return acc.item() / x.shape[0]


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def get_prob(model, images, device):
    logits = model(images.to(device))
    output = softmax(logits.data.cpu().numpy())
    return np.max(output), np.argmax(output)


def reverse_normalize(tensor, mean, std):
    """
    将标准化后的 tensor 转换为原始像素值 [0,255]
    假设 tensor 的 shape 为 (N, C, H, W)，归一化公式为: (x - mean) / std
    反向操作: x * std + mean，再乘以 255
    """
    mean_tensor = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return (tensor * std_tensor + mean_tensor) * 255


def re_normalize(np_img, mean, std):
    """
    将防御模块处理后的 numpy 图像（[0,255]）转换为标准化后的形式
    假设 np_img 的 shape 为 (H, W, C)
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    # 先归一化到 [0,1]，再根据均值和标准差进行归一化
    return (np_img / 255.0 - mean) / std


def main(args):
    try:
        # ✅ 1️⃣ 确保输出目录存在
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            logger.info("创建输出目录: %s", args.output)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('使用设备: %s', device)

        # ✅ 2️⃣ CSV 结果缓存
        csv_data = []
        detection_results = []  # ✅ **缓存每张图片的检测结果，等进度条结束后再打印**
        if args.csv_output:
            csv_dir = os.path.dirname(args.csv_output)
            os.makedirs(csv_dir, exist_ok=True)

        # ✅ 3️⃣ 载入模型
        path_model = args.model_path
        logger.info("正在加载模型: %s", path_model)
        try:
            model = ResNetModel(architecture=args.model_arch, num_classes=10)
            state_dict = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            logger.info("模型加载成功: %s", path_model)
        except Exception as e:
            logger.error("加载模型失败: %s", str(e))
            sys.exit(1)  # 退出程序，错误码为1

        # ✅ 4️⃣ 图像预处理
        SIZE = 299
        MEAN = [0.1307, ]
        STD = [0.3081, ]
        transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        # ✅ 5️⃣ 加载数据集
        try:
            test_dataset = BaseDataset(root=args.dataset, transform=transform)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )
        except Exception as e:
            logger.error("读取数据集失败: %s", traceback.format_exc())
            sys.exit(1)

        # ✅  6️⃣  开始检测
        detected = 0
        num = 0
        start_time = time.time()

        # 进度条
        pbar = tqdm(total=len(test_loader), desc="任务进度", unit="batch", dynamic_ncols=True)

        for batch_data, batch_img_path in test_loader:
            # 遍历当前 batch 中的每个样本
            for i in range(batch_data.size(0)):
                data = batch_data[i].unsqueeze(0)
                img_path = batch_img_path[i]
                name = os.path.basename(img_path)
                save_path = os.path.join(args.output, name)

                try:
                    # ###############################################################################
                    # 编写对抗检测算法，检测对抗样本。
                    # 要求：
                    #   1、检测到的对抗样本数量要累加到 detected 变量里
                    #   2、并将检测到的对抗样本保存到 save_path 路径
                    #   3、将检测结果输保存到 csv_data 列表中，列表的每个元素是一个元组，元组包含文件名（只有文件名，不包含目录）和检测结果（1对抗过，0没对抗过）
                    #       如：（"001.png", 1）
                    #   ❗️❗️❗️    csv_data 中的数据会输出到 csv 文件中用于评分，请务必正确保存。
                    #   4、检测到的信息请保存到 detection_results 列表中，方便后面输出及调试
                    #   
                    # ********************************* 答题区域 ************************************


                    raise NotImplementedError('请补全 /home/work/AdversarialAttackDefense/adversarial-defense.py 脚本中 main() 函数中检测算法的代码; 补全代码后请注释或删除掉此行代码！！！')


                    # ********************************* 答题区域 ************************************
                
                except NotImplementedError as e:
                    logger.error(traceback.format_exc())
                    exit(1)
                except Exception as e:
                    logger.error("处理图像[%s]时出错: %s", name, str(e))
                num += 1

            # **更新进度条**
            pbar.update(1)

        pbar.close()  # **关闭进度条**

        # ✅ **7️⃣ 进度条完成后，再打印所有检测结果**
        for result in detection_results:
            logger.info(result)

        # ✅ **8️⃣ 最后一次性写入 CSV**
        if args.csv_output:
            with open(args.csv_output, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(csv_data)

        # 统计时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("检测完成，总样本数: %d，对抗样本数: %d，耗时: %.2f秒", num, detected, elapsed_time)

        # 计算检测率
        detection_rate = (detected / num) * 100 if num > 0 else 0
        logger.info("对抗样本检测率: %.2f%%", detection_rate)

    except KeyboardInterrupt:
        logger.error("程序被用户中断!")
        sys.exit(0)
    except Exception as e:
        logger.error("程序发生了错误:\n", traceback.format_exc())
        sys.exit(1)


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def print_log(strs, log):
    print(strs)
    log.write(strs)


def evaluate(output, label):
    output = softmax(output)
    pred_idx = np.argmax(output, axis=1)
    acc = np.sum(pred_idx == label) / len(label)
    return acc


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    main(args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    logger.info("程序运行时间：{}小时{}分钟{:.4f}秒".format(hours, minutes, seconds))
