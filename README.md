# 基于PyTorch Lightning上设计的模板代码
⭐ 初衷是为了做实验时尽量的减少些代码量

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
### 1.2 说明
⚙️ 配置文件 
+ ├── config 
  +  ├── base_config.py 
  +  ├── cifar10_resnet.py
  +  └── config_parsing.py