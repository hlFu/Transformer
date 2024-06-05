# Transformer
## Introduction
A PyTorch implementation of vanilla transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Usage
### 1. Training 

Train the model by running train.py. Basic configs are provided in config.py. Configs can be overridden by command-line args.

Example command:
```
python train.py --model_save_path=/path/to/root/transformer/transformer.pt --data_root_path=/path/to/root/data/Multi30k --device=cuda
```

2 basic configs are provided. test_configs are used by default and can be changed to standard_configs in train.py. 
standard_configs is a copy of configs mentioned by the paper.

### 2. Inference
Run predict.py. 

Example command:
```
python predict.py --model_save_path=/Users/fhaolin/work/models/transformer/transformer.pt --sequence="Hello, this is Alex. How are you?" --device=cuda
```