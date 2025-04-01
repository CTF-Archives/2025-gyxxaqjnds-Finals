靶场介绍
对抗生成（Adversarial Generation）
定义

对抗生成是一种通过深度学习技术（尤其是生成对抗网络，GANs）或其他生成方法来制造对抗样本的过程。这些样本通常是针对机器学习模型的特殊输入，旨在欺骗模型，使其做出错误预测或分类。
实际使用场景

    1. 深度学习模型漏洞测试：用于评估神经网络的鲁棒性，测试模型在面对对抗性攻击时的可靠性。

    2. 自动化数据增强：生成对抗样本，使模型在训练时能够学习到更广泛的分布，提高泛化能力。

    3. 隐写术与数据隐匿：用于加密通信或信息隐藏，使数据在视觉或统计上不易被察觉。

    4. 游戏与内容生成：利用对抗生成技术创建更加逼真的图像、视频或音频内容，例如DeepFake技术。

    5. 安全攻击：用于研究对抗性攻击对自动驾驶、医疗AI、金融AI等系统的影响，以提高系统安全性。
技术原理

对抗生成的核心思想是通过优化某种损失函数，使得生成的样本能够最大程度地欺骗目标模型。常见的方法包括：

    1. 生成对抗网络（GANs）：由生成器（Generator）和判别器（Discriminator）组成，通过博弈的方式生成逼真的数据。

    2. 对抗性攻击方法： 

        1. FGSM（Fast Gradient Sign Method）：使用目标模型的梯度信息，快速生成对抗样本。

        2. PGD（Projected Gradient Descent）：在FGSM的基础上进行多步优化，得到更强的对抗样本。

        3. CW（Carlini & Wagner）攻击：通过优化目标函数找到最小扰动，使得模型错误分类。

        4. DeepFool：迭代地寻找最近的决策边界，使输入跨越该边界以欺骗模型。

对抗检测（Adversarial Detection）
定义

对抗检测是指识别对抗样本并区分其与真实数据的过程。其核心目标是保护深度学习模型免受对抗性攻击的影响，提高系统的安全性和稳健性。
实际使用场景

    1. AI安全：检测网络攻击中的恶意输入，提高自动驾驶、医疗诊断等AI系统的防御能力。

    2. 生物识别：防止人脸识别、指纹识别等系统受到对抗攻击的影响，增强身份验证的安全性。

    3. 金融风控：防止恶意操纵的交易数据或欺诈性输入影响信用评分或交易预测模型。

    4. 内容审核：用于检测伪造的DeepFake视频、对抗性修改的文本或图像。
技术原理

对抗检测的核心思想是识别对抗样本与正常样本之间的特征差异，主要的方法包括：

    1. 统计方法： 

        1. 计算数据分布的统计特征，例如均值、方差、熵等，对比对抗样本和正常样本的差异。

    2. 模型不确定性检测： 

        1. 通过贝叶斯神经网络或MC Dropout，评估模型对输入数据的不确定性，高不确定性通常表明样本可能是对抗样本。

    3. 输入变换检测： 

        1. 通过输入去噪（如JPEG压缩、随机变换）观察预测结果的稳定性，如果输入稍作修改后模型输出大幅变化，则可能是对抗样本。

    4. 对抗训练： 

        1. 在训练阶段加入对抗样本，使模型学会识别和抵抗对抗攻击。

    5. 深度特征分析： 

        1. 通过可视化或提取深度神经网络的中间层特征，分析对抗样本和正常样本的不同。

这两个概念在安全领域紧密相关：对抗生成用于制造攻击样本，而对抗检测用于识别并防御这些样本，从而提高AI系统的安全性。


在本任务中，要求参赛者完善对抗检测算法，检测出所有图像中被对抗的图像。



任务要求： 


1、完善 /home/work/AdversarialAttackDefense/adversarial-defense.py 中main() 函数中步骤6的代码。

```

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

                    # ********************************* 答题区域 ************************************



                    raise NotImplementedError('请补全 /home/work/AdversarialAttackDefense/adversarial-defense.py 脚本中 main() 函数中检测算法的代码; 补全代码后请注释或删除掉此行代码！！！')



                    # ********************************* 答题区域 ************************************

                

                except NotImplementedError as e:

                    logger.error(traceback.format_exc())

                    exit(1)

```

2、如误删打靶初始代码，请参考  /home/work/AdversarialAttackDefense/adversarial-defense_backup.py 文件进行恢复。


3、如需在 jupyter 中进行调试，请在终端执行以下步骤：

3.1 获取数据集

```bash

mkdir -p /home/test/ && cd /home/test

wget https://range-datastes.oss-cn-beijing.aliyuncs.com/dk_images.zip

unzip dk_images.zip

```

3.2 执行脚本

```

mkdir -p /tmp/output/

python3 /home/work/AdversarialAttackDefense/adversarial-defense.py --dataset /home/test/images --model_arch res50 --model_path /home/data/mnist_best_res50.pt --csv_output /tmp/output/result.csv --output /tmp/output/images --batch_size 8 --algos feature_squeeze --num_workers  2

```



预安装包：

kornia==0.7.3

natsort==8.4.0

opencv_python_headless==4.10.0.84

Pillow

PyYAML==6.0.2

scripts==3.0

numpy

scipy

scikit-learn

tqdm

torch==2.3.1+cu118 

torchaudio==2.3.1+cu118 

torchvision==0.18.1+cu118 

adversarial-robustness-toolbox

pytorch_msssim

numba

pycocotools

multiprocess




数据集介绍

特别说明：参赛选手不允许使用额外数据

本次靶场使用自定义数据集，该数据集约 550 张图片

数据集格式如下：

    数据集总量约为550张，数据集结构示例如下：

├── data
│ 
 ├── 000001.jpg
 ├── 000002.jpg
 ├──...
 ├── 000050.jpg
 ├──...

 评价指标

在对数据对抗后，会输出一个 csv 文件，文件中记录了所有图像的文件名以及是否被对抗。未对抗标记为0，对抗图像标记为1。

在对抗检测后，也会输出一个 csv 文件，文件中记录了所有图像的文件名以及检测出的是否被对抗。检测结果中的未对抗图像标记为0，对抗图像标记为1。

csv文件中包括两列，内容示例如下：

File				Value

001.jpg		0

002.jpg		1

会根据生成结果和检测结果，计算出选手的TP（True Positive）、TN（True Negative），并计算得到选手的最终准确率。


