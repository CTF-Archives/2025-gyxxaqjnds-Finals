# Active Learning Framework For Deepfake-Detection


## Model Training

### Dataset
`
/your dir/real/files
` <br>
`
/your dir/fake/files
`
### Training
`
CUDA_VISIBLE_DEVICES=0 python3 main.py --consistency cos \
                                        --aug-name RaAug \
                                        --exp-name test \
                                        --epochs 15 \
                                        --load-model-path /previous model path \
                                        --base_xception \
                                        --mode 3
`
### Testing
`
CUDA_VISIBLE_DEVICES=0 python3 main.py --consistency cos \
                                        --aug-name RaAug \
                                        --exp-name test \
                                        --epochs 15 \
                                        --load-model-path /previous model path \
                                        --base_xception \
                                        --mode 3 \
                                        --test true
`

## Active Query

### Unlabeled Dataset
`python3 query.py --consistency cos 
                --fake_type NeuralTextures 
                --root /your unlabeled data path 
                --exp-name test 
                --load-model-path /your model path 
                --output /存储路径
                --num 筛选数量 
                --query_type CLUE
`

`unlabeled.json` 存储未被选择的图像文件名
`tolabeled.json` 存储被选择的图像文件名