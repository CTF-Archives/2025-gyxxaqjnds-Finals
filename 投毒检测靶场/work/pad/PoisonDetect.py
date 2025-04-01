import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from log_utils import setup_logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import transforms
from resnet18_Detection import resnet18Detection

logger = setup_logger()

DEFENSE_MODE = {'activation', 'io_relation', "resnet18"}


class CustomArgumentParser(argparse.ArgumentParser):
    """自定义参数解析器，将错误信息记录到日志"""

    def error(self, message):
        logger.error(f"参数解析错误: {message}")
        self.exit(2, f"ERROR: {message}")  # 保留红色错误提示，但内容更简洁


# 解析命令行参数
def parse_args():
    logger.info("开始解析参数")
    parser = CustomArgumentParser(
        description='毒害检测.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--detection_method', type=str, help='检测方法', choices=DEFENSE_MODE)
    parser.add_argument('--model_path', type=str, help='模型目录')
    parser.add_argument('--clean_data_path', type=str, default='./clean_data.npy', help='干净数据路径')
    parser.add_argument('--poisoned_data_path', type=str, default='./poisoned_data.npy', help='污染数据路径')
    parser.add_argument('--layer_name', type=int, default=2, help='层名称')
    parser.add_argument('--threshold', type=float, default=9,
                        help='用于 io_relation 检测或激活检测的阈值io_relation建议设置为9,activation建议设置为30')
    parser.add_argument('--output_csv', type=str, default='/tmp/result.csv', help='result.csv')
    parser.add_argument('--output', type=str, default='/tmp/result', help='output')

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # 防止二次输出错误信息，直接退出
        sys.exit(e.code)
    logger.info(f"命令行参数解析完成: {args}")
    return args


class BackdoorDetectionModule:
    def __init__(self, model_class):
        """
        初始化模块，传入模型类以便后续加载模型。

        :param model_class: 定义模型结构的类
        """
        self.model_class = model_class
        logger.info("初始化检测模块")

    def load_npy_dataset(self, npy_file, batch_size=32):
        """
        加载数据集（从npy文件中加载），并返回小批量数据的生成器。

        :param npy_file: npy文件的路径
        :param batch_size: 每个小批量的数据大小
        :return: 数据生成器，返回每个小批量的图像 Tensor
        """
        logger.info(f"加载数据集: {npy_file}")

        # 检查文件是否存在
        if not os.path.exists(npy_file):
            logger.error(f"文件路径不存在: {npy_file}")
            raise FileNotFoundError(f"文件路径不存在: {npy_file}")

        try:
            dataset = np.load(npy_file)
        except Exception as e:
            logger.error(f"加载数据时发生错误: {e}")
            raise

        # 如果数据是从0-255的像素值，则归一化到0-1
        if dataset.max() > 1.0 and dataset.dtype != np.float32:
            dataset = dataset.astype(np.float32) / 255.0

        # 如果数据需要从形状 (N, H, W, C) 转换到 (N, C, H, W)
        if len(dataset.shape) == 4 and dataset.shape[-1] == 3:
            dataset = np.transpose(dataset, (0, 3, 1, 2))

        dataset = torch.tensor(dataset)

        # 分批次加载数据
        def batch_loader():
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i + batch_size]

        logger.info(f"成功加载了 {dataset.size(0)} 张图片")
        return batch_loader()

    def load_dataset(self, data_dir, batch_size=32):
        """
        从指定目录加载图像数据集，返回分批次的数据生成器。

        :param data_dir: 包含图像文件的目录路径（支持子目录）
        :param batch_size: 每个批次的数据量
        :return: 分批次的数据生成器
        """
        logger.info(f"加载数据集目录: {data_dir}")

        logger.info(f"开始加载用户自定义数据集")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像大小
            transforms.ToTensor()  # 转换为张量
        ])

        # 否则从文件夹加载数据
        x_train = []
        y_train = []
        filenames = []
        # 遍历目录，假设目录结构为 ./label1/img1.jpg, ./label1/img2.jpg, ./label2/img1.jpg, ...
        for label_name in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_name)
            if os.path.isdir(label_path):
                # 遍历该标签下的所有图像文件
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')  # 加载图片并转换为RGB格式
                        img_tensor = transform(img)  # 应用变换（调整大小并转换为张量）
                        x_train.append(img_tensor)
                        y_train.append(int(label_name))  # 标签名直接作为类别标签
                        filenames.append(img_name)  # 保存原始文件名
                    except Exception as e:
                        logger.warning(f"无法加载图片 {img_path}: {e}")

        # 转换为张量堆叠
        dataset_tensor = torch.stack(x_train)

        # 创建批次生成器
        def batch_loader():
            for i in range(0, len(dataset_tensor), batch_size):
                yield dataset_tensor[i:i + batch_size]

        # 创建标签批次生成器
        def label_loader():
            for i in range(0, len(y_train), batch_size):
                yield y_train[i:i + batch_size]

        # 创建文件名批次生成器
        def filenames_loader():
            for i in range(0, len(filenames), batch_size):
                yield filenames[i:i + batch_size]

        logger.info(f"成功加载 {len(dataset_tensor)} 张图像")
        return batch_loader(), label_loader(), filenames_loader()

    def detect_poisoning(self, model, clean_data_loader, poisoned_data_loader, method='activation', layer_name=None,
                         threshold=0.9, output_csv='/tmp/result.csv', output='/tmp/output', label_loader=None,
                         filnames_loader=None):
        """
        集成的后门检测算法，包含激活模式分析和输入-输出关系检测。

        :param model: 训练好的模型
        :param clean_data_loader: 干净数据的小批量生成器
        :param poisoned_data_loader: 有毒数据的小批量生成器
        :param method: 'activation' 或 'io_relation'
        :param layer_name: 在激活模式分析中指定的中间层名称
        :param threshold: 基于输入-输出关系检测中的置信度阈值,同时也用于激活模式分析中的阈值
        :return: 检测结果
        """
        if threshold < 0:
            logger.error("置信度阈值不能为负数")
            raise ValueError("置信度阈值不能为负数")

        logger.info(f"开始执行后门检测方法: {method}")
        if method == 'activation':
            if layer_name is None:
                logger.error("激活模式分析需要指定中间层名称")
                raise ValueError("激活模式分析需要指定中间层名称")
            self._activation_pattern_analysis(model, clean_data_loader, poisoned_data_loader, layer_name, threshold,
                                              output_csv, output, label_loader, filnames_loader)
        elif method == 'io_relation':
            abnormal_samples = self._input_output_relation_detection(model, clean_data_loader, poisoned_data_loader,
                                                                     threshold, output_csv, output, label_loader,
                                                                     filnames_loader)
            self._plot_io_relation(abnormal_samples)
        else:
            logger.error("无效的检测方法, 请选择 'activation' 或 'io_relation'")
            raise ValueError("无效的检测方法选择，请选择 'activation' 或 'io_relation'")

    def _plot_io_relation(self, abnormal_samples):
        """
        可视化输入-输出关系检测的结果。
        """
        plt.figure()
        labels = ['Normal', 'Abnormal']
        counts = [32 - abnormal_samples, abnormal_samples]  # 32为假设批量大小
        plt.bar(labels, counts, color=['blue', 'red'])
        plt.xlabel('Sample Type')
        plt.ylabel('Count')
        plt.title('Input-Output Relation Detection Result')
        plt.savefig('./io_relation_result.png')
        logger.info("输入-输出关系检测结果已保存为图片: ./io_relation_result.png")
        plt.close()

    def _activation_pattern_analysis(self, model, clean_data_loader, poisoned_data_loader, layer_name, threshold, output_csv, output,label_loader,filnames_loader):
        """_summary_

        Args:
            model (_type_): 传入加载好的模型
            clean_data_loader (_type_): 用于加载干净数据集
            poisoned_data_loader (_type_): 用于加载投毒的数据集
            layer_name (_type_): 传入中间层名称
            threshold (_type_): 阈值，用于判断每个样本是否有害
            output_csv (_type_): 输出csv文件的路径
            output (_type_): 输出结果文件的路径
            label_loader (_type_): 创建结果的标签文件
            filnames_loader (_type_): 加载原有的图片文件名
        """
        logger.info(f"开始执行激活模式分析: {layer_name}")

        intermediate_layer_model = torch.nn.Sequential(*list(model.children())[:layer_name])

        clean_activations, poisoned_activations = [], []
        poisoned_images = []  # 保存异常样本的图片
        with torch.no_grad():  # 禁用梯度计算，减少内存占用
            for clean_batch in clean_data_loader:
                clean_act = intermediate_layer_model(clean_batch).detach().numpy()
                clean_activations.append(clean_act)

            for poisoned_batch in poisoned_data_loader:
                poisoned_act = intermediate_layer_model(poisoned_batch).detach().numpy()
                poisoned_activations.append(poisoned_act)

                poisoned_images.append(poisoned_batch)

        clean_activations = np.concatenate(clean_activations)
        poisoned_activations = np.concatenate(poisoned_activations)

        # PCA 降维和可视化
        pca = PCA(n_components=2)
        clean_activations_2d = pca.fit_transform(clean_activations.reshape(len(clean_activations), -1))
        poisoned_activations_2d = pca.transform(poisoned_activations.reshape(len(poisoned_activations), -1))

        # 将所有数据合并，准备聚类
        all_activations_2d = np.concatenate([clean_activations_2d, poisoned_activations_2d], axis=0)

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=2)  # 假设聚成两类，正常样本和带毒样本
        kmeans.fit_predict(all_activations_2d)

        # 获取每个聚类的大小
        cluster_sizes = np.bincount(kmeans.labels_)

        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_

        # 计算每个样本到聚类中心的距离
        distances = pairwise_distances_argmin_min(poisoned_activations_2d, cluster_centers)[1]

        # 用于创建.csv文件
        sample_labels, filenames = [], []

        # for distance in distances:
        #     if distance < threshold:
        #         sample_labels.append(1)  # 带毒样本
        #     else:
        #         sample_labels.append(0)  # 无毒样本

        # 创建防御结果的文件夹
        defense_result = output
        os.makedirs(defense_result, exist_ok=True)
        # 创建保存干净图片的文件夹
        clean_images_dir = defense_result + "/clean_images"
        os.makedirs(clean_images_dir, exist_ok=True)

        # 创建保存带毒图片的文件夹
        poisoned_images_dir = defense_result + "/poisoned_images"
        os.makedirs(poisoned_images_dir, exist_ok=True)

        current_index = 0  # 跟踪距离数组的起始索引
        result_rows = []
        for poisoned_batch, label_batch, filenames_batch in zip(poisoned_images, label_loader, filnames_loader):
            batch_size = len(poisoned_batch)
            batch_distances = distances[current_index:current_index + batch_size]
            for img_tensor, label, filename, distance in zip(poisoned_batch, label_batch, filenames_batch,
                                                             batch_distances):
                filenames.append(filename)  # 无论是否带毒，都生成文件名

                if distance < threshold:
                    # 创建对应的标签文件夹
                    poisoned_label_dir = os.path.join(poisoned_images_dir, str(label))
                    os.makedirs(poisoned_label_dir, exist_ok=True)

                    sample_labels.append(1)  # 带毒样本
                    result_rows.append([filename, 1])  # 带毒样本
                    # 保存图片
                    img = img_tensor.permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    # print(filename)
                    img.save(os.path.join(poisoned_label_dir, filename.split('\\')[-1]))
                else:
                    # 创建对应的标签文件夹
                    clean_label_dir = os.path.join(clean_images_dir, str(label))
                    os.makedirs(clean_label_dir, exist_ok=True)

                    result_rows.append([filename, 0])  # 无毒样本
                    sample_labels.append(0)  # 无毒样本
                    img = img_tensor.permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(os.path.join(clean_label_dir, filename.split('\\')[-1]))
                # image_index += 1
            current_index += batch_size

        # 将结果保存到 CSV 文件
        result_data = {
            "Filename": filenames,
            "Poisoned": sample_labels
        }

        save_results_to_csv(result_rows, output_csv)
        logger.info(f"结果已保存为CSV: {output_csv}")

        # 画图并保存
        plt.figure()
        plt.scatter(clean_activations_2d[:, 0], clean_activations_2d[:, 1], c='blue', label='Clean')
        plt.scatter(poisoned_activations_2d[:, 0], poisoned_activations_2d[:, 1], c='red', label='Poisoned')
        plt.legend()

        # 保存图片到指定路径
        plt.savefig('./activation_result.png')
        logger.info("聚类结果已保存为图片: ./activation_result.png")
        plt.close()  # 关闭图窗口，防止显示


    def _input_output_relation_detection(self, model, clean_data_loader, poisoned_data_loader, threshold=0.9, output_csv='/tmp/result.csv', output='/tmp/output',label_loader=None,filnames_loader=None):
        """_summary_

        Args:
            model : 模型
            clean_data_loader :  干净数据集的加载器
            poisoned_data_loader :  投毒数据集的加载器
            threshold (float, optional): 阈值 ，如果某个投毒样本的最大预测置信度大于阈值则视为异常样本 
            output_csv (str, optional):  输出csv文件的路径
            output (str, optional):  输出结果文件夹的路径
            label_loader ： 用于结果创建标签文件夹
            filnames_loader (_type_, optional):  用于加载原有图片名字

        Returns:
            int : 检测为异常的样本数量
        """
        logger.info(f"开始执行输入-输出关系检测")
        results = []  # 用于生成最后的.csv结果文件
        abnormal_samples = 0
        # count = 0 # 用于对编号进行计数

        # 创建防御结果的文件夹
        defense_result = output
        os.makedirs(defense_result, exist_ok=True)
        # 创建保存干净图片的文件夹
        clean_images_dir = defense_result + "/clean_images"
        os.makedirs(clean_images_dir, exist_ok=True)
        # 创建保存带毒图片的文件夹
        poisoned_images_dir = defense_result + "/poisoned_images"
        os.makedirs(poisoned_images_dir, exist_ok=True)

        with torch.no_grad():  # 禁用梯度计算，减少内存占用
            for clean_batch, poisoned_batch, label_batch, filenames_batch in zip(clean_data_loader,
                                                                                 poisoned_data_loader, label_loader,
                                                                                 filnames_loader):
                clean_predictions = model(clean_batch).detach().numpy()
                poisoned_predictions = model(poisoned_batch).detach().numpy()
                poisoned_confidence = np.max(poisoned_predictions, axis=1)
                abnormal_samples += np.sum(poisoned_confidence > threshold)

                # 假设每个批次的样本可以根据其在文件中的位置生成文件名
                for image,label,filename,confidence in zip(poisoned_batch, label_batch ,filenames_batch, poisoned_confidence):
                    is_poisoned = 1 if confidence > threshold else 0
                    # 将文件名和标记保存到结果列表中
                    results.append((filename, is_poisoned))

                    if is_poisoned == 1:
                        # 创建标签文件夹
                        poisoned_label_dir = os.path.join(poisoned_images_dir, str(label))
                        os.makedirs(poisoned_label_dir, exist_ok=True)

                        # 保存毒害样本图片
                        image = image.permute(1, 2, 0).numpy()
                        image = (image * 255).astype(np.uint8)
                        image = Image.fromarray(image)
                        image.save(os.path.join(poisoned_label_dir, filename.split('\\')[-1]))
                    else:
                        clean_label_dir = os.path.join(clean_images_dir, str(label))
                        os.makedirs(clean_label_dir, exist_ok=True)
                        # 保存干净样本图片
                        image = image.permute(1, 2, 0).numpy()
                        image = (image * 255).astype(np.uint8)
                        image = Image.fromarray(image)
                        image.save(os.path.join(clean_label_dir, filename.split('\\')[-1]))
                    # count += 1

        logger.info(f"检测到 {abnormal_samples} 个异常样本")
        save_results_to_csv(results, output_csv)
        return abnormal_samples

    def _save_results_to_csv(self, results):
        """
        保存检测结果到 CSV 文件
        """
        with open('detection_io_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Poisoned'])
            writer.writerows(results)
        logger.info("检测结果已保存为 'detection_io_results.csv'")


def save_results_to_csv(results, output_csv):
    """
    保存检测结果到 CSV 文件
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    logger.info(f"检测结果已保存为 '{output_csv}'")


# 实例化检测模块
def main(config):
    if config["detection_method"] == 'activation' or config["detection_method"] == 'io_relation':
        # 这里修改了一下原来的设置为config['model_path']
        # 检查模型文件路径是否存在
        if not os.path.exists(config['model_path']):
            logger.error(f"模型文件不存在: {config['model_path']}")
            raise FileNotFoundError(f"模型文件不存在: {config['model_path']}")
        # 加载模型
        logger.info(f"开始加载模型: {config['model_path']}")
        # 加载模型
        try:
            model = torch.load(config['model_path'], weights_only=False)
            model.eval()
            logger.info(f"成功加载模型: {config['model_path']}")
        except Exception as e:
            logger.error(f"加载模型时发生错误: {e}")
            raise e

        # 实例化检测模块
        logger.info("实例化 BackdoorDetectionModule 模块...")
        detection_module = BackdoorDetectionModule(None)

        # 加载数据集（使用分批加载器）
        # 加载干净数据集
        logger.info(f"加载干净数据集: {config['clean_data_path']}")
        clean_data_loader, _, _ = detection_module.load_dataset(config['clean_data_path'], batch_size=32)
        logger.info("干净数据集加载成功。")

        # 加载带毒数据集
        logger.info(f"加载带毒数据集: {config['poisoned_data_path']}")
        poisoned_data_loader, poisoned_label_loader, poisoned_filenames_loader = detection_module.load_dataset(
            config['poisoned_data_path'], batch_size=32)
        logger.info("带毒数据集加载成功。")

        # 执行后门检测
        logger.info(f"检测方法: {config['detection_method']}")
        logger.info(f"检测层名称: {config['layer_name']} (适用于激活模式分析)")
        logger.info(f"阈值: {config['threshold']}")
        logger.info(f"输出结果目录: {config['output']}")
        logger.info(f"结果保存路径 (CSV): {config['output_csv']}")

        # 执行后门检测
        detection_module.detect_poisoning(
            model=model,
            clean_data_loader=clean_data_loader,
            poisoned_data_loader=poisoned_data_loader,
            method=config['detection_method'],
            layer_name=config['layer_name'],
            threshold=config['threshold'],
            output_csv=config['output_csv'],
            output=config['output'],
            label_loader=poisoned_label_loader,
            filnames_loader=poisoned_filenames_loader
        )
        logger.info("数据投毒检测完成")
    elif config["detection_method"] == 'resnet18':
        resnet18Detection(config['poisoned_data_path'], config['output_csv'], config['threshold'], config['output'])
    else:
        raise Exception("Not supported detection method")


if __name__ == "__main__":
    args = parse_args()
    # 将命令行参数转换为字典，并加载到配置中
    config = {
        "model_path": args.model_path,
        "clean_data_path": args.clean_data_path,
        "poisoned_data_path": args.poisoned_data_path,
        "detection_method": args.detection_method,
        "layer_name": args.layer_name,
        "threshold": args.threshold,
        "output_csv": args.output_csv,
        "output": args.output
    }

    logger.info(f"最终配置: {json.dumps(config, indent=4)}")

    # 执行主程序
    main(config)
