任务说明

模型投毒（Model Poisoning）和投毒检测（Poisoning Detection）是机器学习安全领域的两个关键方向。它们在实际使用场景中具有重要的意义，尤其是在涉及敏感数据、关键决策和高安全性要求的领域。

模型投毒是一种对抗性攻击方法，攻击者通过操纵训练数据或模型更新过程，破坏模型的性能或植入后门。模型投毒的实际使用场景包括垃圾邮件过滤系统、人脸识别系统、推荐系统、医疗诊断系统、自动驾驶系统等。投毒检测主要用于识别和防御模型投毒攻击，确保模型的可靠性和安全性。

为此，开设模型投毒和投毒检测的靶场，该靶场可以为学习人员、研究人员和安全从业者提供一个标准化的实验平台，支持对投毒攻击和防御方法的深入研究。通过靶场，用户可以更好地理解模型投毒的原理、方法和防御措施，从而提高机器学习模型的安全性和鲁棒性。

在靶场中，参赛人员或队伍在给定的投毒数据集上进行算法研发，同时会对算法效果进行评估。

比赛任务

在本任务中，要求参赛者设计投毒检测算法，检测出所有图像中被投毒的图像。

任务要求： 

1、使用基于 resnet18 的异常数据检测算法，补全 /home/work/pad/resnet18_Detection.py 文件中 filter_poisoned_samples 函数中缺失的部分。

```python3

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

		...

```


2、打靶时如误删初始代码，可参考 /home/work/pad/PoisonDetect_backup.py 文件进行恢复。


3、如需在 jupyter 中进行调试，请在终端执行以下步骤：

3.1 获取数据集

```shell

mkdir -p /home/test/  && cd /home/test/ 

wget https://range-datastes.oss-cn-beijing.aliyuncs.com/td_images.zip

unzip td_images.zip

```

3.2 执行脚本

```

mkdir -p /tmp/output

cd /home/work/pad 

python3 PoisonDetect.py --output /tmp/output/res --detection_method resnet18 --threshold 0.2 --poisoned_data_path /home/test/images --output_csv /tmp/output/result.csv

```


预安装 Python 包

实验环境预安装了以下 Python 包：

annotated-types==0.7.0

anyio==4.4.0

argon2-cffi==23.1.0

argon2-cffi-bindings==21.2.0

blessed==1.20.0

certifi==2024.8.30

cffi==1.17.1

charset-normalizer==3.3.2

exceptiongroup==1.2.2

fastapi==0.114.0

filelock==3.16.0

fonttools==4.53.1

fsspec==2024.9.0

gpustat==1.1.1

h11==0.9.0

httptools==0.1.2

idna==3.8

importlib-resources==6.4.4

jinja2==3.1.4

joblib==1.4.2

kiwisolver==1.4.7

MarkupSafe==2.1.5

matplotlib==3.7.5

minio==7.2.8

mpmath==1.3.0

networkx==3.1

numpy==1.24.4

nvidia-cublas-cu12==12.1.3.1

nvidia-cuda-cupti-cu12==12.1.105

nvidia-cuda-nvrtc-cu12==12.1.105

nvidia-cuda-runtime-cu12==12.1.105

nvidia-cudnn-cu12==9.1.0.70

nvidia-cufft-cu12==11.0.2.54

nvidia-curand-cu12==10.3.2.106

nvidia-cusolver-cu12==11.4.5.107

nvidia-cusparse-cu12==12.1.0.106

nvidia-ml-py==12.560.30

nvidia-nccl-cu12==2.20.5

nvidia-nvjitlink-cu12==12.6.68

nvidia-nvtx-cu12==12.1.105

opencv-python-headless==4.10.0.84

packaging==24.1

pillow==10.4.0

psutil==6.0.0

pycparser==2.22

pycryptodome==3.20.0

pydantic==2.9.1

pydantic-core==2.23.3

pyparsing==3.1.4

python-dateutil==2.9.0.post0

requests==2.32.3

scikit-learn==1.3.2

scipy==1.10.1

six==1.16.0

sniffio==1.3.1

starlette==0.38.5

sympy==1.13.2

threadpoolctl==3.5.0

torch==2.4.1

torchvision==0.19.1

triton==3.0.0

typing-extensions==4.12.2

unicorn==2.0.1

urllib3==2.2.2

uvicorn==0.11.3

uvloop==0.20.0

waitGPU==0.0.3

wcwidth==0.2.13

websockets==8.1

wsproto==0.15.0

zipp==3.20.1

tqdm==4.67.1

pandas==2.0.3

数据集介绍

特别说明：参赛选手不允许使用额外数据

本次靶场使用自定义数据集，该数据集是 cifar-10 数据集的子集。

数据集格式如下：

    数据集总量为200，并按标签分类，数据集结构示例如下：

├── data
│ 
│├── 0
 │ ├── 000001.jpg
 │ ├── 000002.jpg
 │ ├──...
 │ └── 000020.jpg
 ├── 1   
 │ ├── 000001.jpg
 │ ├── 000002.jpg
 │ ├──...
 │ └── 000020.jpg
 ├──...
 └── 9   
     ├── 000001.jpg
     ├── 000002.jpg
     ├──...
     └── 000020.jpg


评价指标

在对数据投毒后，会输出一个 csv 文件，文件中记录了所有图像的文件名以及是否被投毒。未投毒标记为0，投毒图像标记为1。

在投毒检测后，也会输出一个 csv 文件，文件中记录了所有图像的文件名以及检测出的是否被投毒。检测结果中的未投毒图像标记为0，投毒图像标记为1。

csv文件中包括两列，内容示例如下：

File				Value

0/001.jpg		0

0/002.jpg		1

会根据投毒结果和检测结果，计算出选手的TP（True Positive）、TN（True Negative），并计算得到选手的最终准确率。

