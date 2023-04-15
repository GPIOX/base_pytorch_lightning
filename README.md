<!--
 * @Descripttion: 
 * @version: 
 * @Author: Cai Weichao
 * @Date: 2023-04-14 21:13:38
 * @LastEditors: Cai Weichao
 * @LastEditTime: 2023-04-15 17:15:30
-->
# 基于PyTorch Lightning上设计的模板代码
⭐ 参考了MMSegmentation的项目配置，初衷是为了做实验时尽量的减少些代码量

## 1. 文件构成and说明
### 1.1 文件构成
```bash
├── config
│   ├── base_config.py
│   ├── cifar10_resnet.py
│   └── config_parsing.py
├── dataset
│   ├── cifar_dataset.py
│   ├── dataloader.py
│   ├── __init__.py
│   └── utils.py
├── main.py
├── img
├── model
│   ├── base_loss.py
│   ├── base_trainer.py
│   ├── __init__.py
│   ├── network.py
│   ├── pretrained
│   └── resnet
│       ├── resnet.py
│       └── trainer.py
├── README.md
└── tools
    └── utils.py
```
具体见图
<div align=center>
  <img src=./img/img2.png>
</div> 


### 1.2 说明
⚙️ 配置文件 
```bash
├── config
│   ├── base_config.py
│   ├── cifar10_resnet.py
│   └── config_parsing.py
```
+ `base_config.py`: 参考了MMSegmentation，一切的初始化，定义了所有的初始化参数
+ `cifar10_resnet.py`: 继承了`base_config.py`，通过变量 _base_ 继承
+ `config_parsing.py`: mmcv中读取`.py`文件里的参数并处理的功能实现

💾 数据集
```bash
├── dataset
│   ├── cifar_dataset.py
│   ├── dataloader.py
│   ├── __init__.py
│   └── utils.py
```
+ `cifar_dataset.py`: 自定义的读取CIFAT-10/100的dataset类
+ `dataloader.py`: 文件里只有一个函数 get_dataloader ，基于传入的参数获取指定的 datset 并制成dataloader，同时加入 torchvision.transforms (可以在配置中指定)
  + 该函数允许自定义行为与传入的参数，只要返回值为 torch.utils.data.DataLoader 即可
+ `__init__.py`: 支持的dataset，为了可以减少改动并自动令 get_dataloader 识别，在自定义完 dataset 类别后，在该文件import，并把类名添加至 \_\_all__ 变量中
+ `utils.py`: 下载cifar数据集用

🤖 模型
```bash
├── model
│   ├── base_loss.py
│   ├── base_trainer.py
│   ├── __init__.py
│   ├── network.py
│   ├── pretrained
│   └── resnet
│       ├── resnet.py
│       └── trainer.py
```
+ `base_loss.py`: 该文件定义了 CustomLoss 这个类。
  + 该类继承自 nn.Module ，允许自定义行为与传入的参数，最终返回值是损失值即可，可以通过配置文件实现配置更多的损失函数（类名可以不用更改）
+ ``
+ ``
+ ``
+ ``
+ ``
