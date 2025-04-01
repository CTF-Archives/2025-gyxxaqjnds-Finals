import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os
import logging
import tqdm
import csv
import argparse
from collections import defaultdict
from log_utils import setup_logger

# 设置日志
logger = setup_logger()


# 自定义数据集类
class CustomDataset(TensorDataset):
    def __init__(self, x, y, filenames):
        super().__init__(x, y)
        self.filenames = filenames

    def __getitem__(self, index):
        return super().__getitem__(index) + (self.filenames[index],)

# ResNet18 封装类
class ResNet18Detector:
    def __init__(self, num_classes=10):
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def load_custom_dataset(self, image_dir, transform):
        x, y, filenames = [], [], []
        for label_name in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_name)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img)
                        x.append(img_tensor)
                        y.append(int(label_name))
                        filenames.append(img_name)
                    except Exception as e:
                        logger.warning(f"无法加载图片 {img_path}: {e}")

        if not x:
            logger.error("没有成功加载任何图片")
            return None

        x = torch.stack(x)
        y = torch.tensor(y, dtype=torch.long)
        logger.info(f"成功加载 {len(x)} 张图片")
        return CustomDataset(x, y, filenames)

    def train_model(self, train_loader, num_epochs=5):
        self.model.train()
        losses_history = defaultdict(list)  # 使用字典，键为文件名

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels, filenames in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 记录每个样本的 Loss 值（按文件名）
                with torch.no_grad():
                    for j in range(inputs.size(0)):
                        filename = filenames[j]
                        sample_loss = self.criterion(outputs[j].unsqueeze(0), labels[j].unsqueeze(0))
                        losses_history[filename].append(sample_loss.item())

        return losses_history

    @staticmethod
    def filter_poisoned_samples(losses_history, all_filenames, threshold_ratio=0.1):
        """
        计算每个样本的 loss 变化率，然后排序，将前 threshold_ratio 比例的样本视为有毒样本

        ：param losses_history：所有样本的损失值（按文件名）
        ：param all_filenames：所有文件名称列表
        ：params threshold_ratio：筛选带毒样本的阈值比例，由命令行传入，可调整
        """

        ##########################################################################
        #
        # TODO:请补全此处代码
        # 代码功能：                                                                    
        #     检测有毒样本，将有毒样本文件名保存到变量 poisoned_filenames 里
        # 要求：
        #     1、在计算过程中，将中间结果保存到以下变量中，用于后续计算或输出
        #        poisoned_filenames：检测出的有毒样本文件名列表
        #

        raise NotImplementedError("请补全 /home/work/pad/resnet18_Detection.py 文件中 filter_poisoned_samples 函数中的代码; 补全代码后请注释或删除掉此行代码！！！")


        #########################################################################


        # 确保所有加载的文件名都被处理
        results = []
        for filename in all_filenames:
            is_poisoned = 1 if filename in poisoned_filenames else 0
            results.append((filename, is_poisoned))
        return results

    @staticmethod
    def save_results_to_csv(samples, output_file):
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(["Filename", "Is_Poisoned"])
            for filename, is_poisoned in samples:
                writer.writerow([filename, is_poisoned])
        logger.info(f"结果已保存到 {output_file}")


# 主函数
def resnet18Detection(image_dir, output_file, threshold_ratio, output):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    detector = ResNet18Detector()
    dataset = detector.load_custom_dataset(image_dir, transform)
    if dataset is None:
        return
    
    # 计算数据集的总样本数
    total_samples = len(dataset)

    # 设置 batch_size，确保每个样本在每个 epoch 中被加载
    batch_size = min(32, total_samples)  # 32 是一个常见的 batch_size，但不超过总样本数

    train_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False)

    losses_history = detector.train_model(train_loader, num_epochs=5)

    all_filenames = dataset.filenames
    all_samples = detector.filter_poisoned_samples(losses_history, all_filenames, threshold_ratio=threshold_ratio)
    detector.save_results_to_csv(all_samples, output_file)

    # 根据检测结果保存图片到对应的文件夹
    os.makedirs(output, exist_ok=True)
    clean_images_dir = os.path.join(output, "clean_images")
    poisoned_images_dir = os.path.join(output, "poisoned_images")
    os.makedirs(clean_images_dir, exist_ok=True)
    os.makedirs(poisoned_images_dir, exist_ok=True)

    for (filename, is_poisoned), label in zip(all_samples, dataset.tensors[1]):
        img_path = os.path.join(image_dir, str(label.item()), filename)
        if is_poisoned:
            target_dir = os.path.join(poisoned_images_dir, str(label.item()))
        else:
            target_dir = os.path.join(clean_images_dir, str(label.item()))
        
        os.makedirs(target_dir, exist_ok=True)
        try:
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(target_dir, filename))
        except Exception as e:
            logger.warning(f"无法保存图片 {img_path}: {e}")

    logger.info(f"图片已根据检测结果保存到 {output}")


def main(config):
    resnet18Detection(config["image_dir"], config["output_file"], config["threshold_ratio"],config["output"])


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="检测带毒样本")
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 CSV 文件路径")
    parser.add_argument("--threshold_ratio", type=float, required=True, help="筛选带毒样本的阈值比例")
    parser.add_argument("--output", type=str, required=True, help="输出文件夹路径")
    args = parser.parse_args()

    config = {
        "image_dir": args.image_dir,
        "output_file": args.output_file,
        "threshold_ratio": args.threshold_ratio,
        "output": args.output
    }
    # 调用主函数
    main(config)