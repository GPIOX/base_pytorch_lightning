pl_trainer_func = dict(
    max_epochs=250,
    accelerator='gpu',
    devices=1,
    precision=32,  # 精度
    # logger = logger # 日志管理器
    val_check_interval=1.0, # 设置训练完一个epoch后才开始验证
)

model = dict(
    name='ResNet18',
    model_arg=dict(
        input_channel=3,
        n_outputs=10,
    ),

    # 设置训练模型相关的参数，如优化器、学习率等
    trainer=dict(
        epoch=pl_trainer_func['max_epochs'],
        device=pl_trainer_func[
            'devices'],  # devices > 1时自定义的trainer中用来对logger操作用
        name='ResNetTrainer',
        trainer_arg=dict(
            # 定义损失函数
            loss=dict(
                name='CustomLoss',
                weight=None,
                loss_1=dict(
                    name='CrossEntropyLoss',
                    loss_arg = dict(reduce=False, ignore_index=-1),
                ),

            ),
            # 定义优化器
            optimizer=dict(name='SGD',
                           optimizer_arg=dict(
                               lr=0.01,
                               momentum=0.9,
                               weight_decay=5E-4,
                           )),

            # 定义学习率调整
            lr_scheduler=dict(
                name='MultiStepLR',
                scheduler_arg=dict(
                    milestones=[
                        int(pl_trainer_func['max_epochs'] * 0.25),
                        int(pl_trainer_func['max_epochs'] * 0.5),
                        int(pl_trainer_func['max_epochs'] * 0.75),
                    ],
                    gamma=0.5,
                ),
            ),
        ),
    ),
)

# 定义训练集相关参数
train_dataset = dict(
    name='CIFAR10',
    dataset_arg=dict(
        root='./data',
        download=True,
        train=True,
        noise_type='pairflip',
        noise_rate=0.2,
    ),
    dataloader_arg=dict(
        batch_size=128,
        num_workers=4,
        shuffle=True,
        # pin_memory = true
    ),
    transform=dict(
        RandomCrop=dict(
            size=32,
            padding=4,
        ),
        RandomHorizontalFlip=None,
        ToTensor=None,
    ),
)

# 定义测试集相关参数
test_dataset = dict(
    name='CIFAR10',
    dataset_arg=dict(
        root='./data',
        download=True,
        train=False,
        noise_type='pairflip',
        noise_rate=0.2,
    ),
    dataloader_arg=dict(
        batch_size=128,
        num_workers=4,
        shuffle=False,
        # pin_memory = true
    ),
    transform=dict(ToTensor=None, ),
)

# 定义断点保存
callbacks = dict(
    ModelCheckpoint1_arg=dict(
        save_top_k=1,
        monitor="valid_loss",
        filename='{epoch}-{valid_loss:.3f}',
    ),
    ModelCheckpoint2_arg=dict(
        save_top_k=1,
        monitor="valid_acc",
        filename='{epoch}-{valid_acc:.3f}',
        mode="max",
    ),
)

# 定义logger
logger = dict(
    logger1=dict(
        name='CSVLogger',
        logger_arg=dict(
            save_dir="experment_logs",
            name=f"CSVLog",
            flush_logs_every_n_steps=1
        ),
    ),
    logger2=dict(
        name='TensorBoardLogger',
        logger_arg=dict(
            save_dir="experment_logs",
            name=f"TensorBoardLog",
        ),
    ),
)