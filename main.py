import pytorch_lightning as pl
import argparse
import os

from dataset.dataloader import get_dataloader
from model.network import get_trainer
from tools.utils import get_checkpoint_callpack, get_logger
from config.config_parsing import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config',
                        help='train config file path',
                        default='config/cifar10_resnet.py')
    parser.add_argument('--ckpt',
                        help='train config file path',
                        default=None)

    args = parser.parse_args()

    return args


"""
    _summary_: train and test start
"""
def main(args):
    # 读取配置文件

    config = Config.fromfile(args['config'])

    # 定义dataloader
    train_dataloader = get_dataloader(**config['train_dataset'])
    test_dataloader = get_dataloader(**config['test_dataset'])

    if 'valid_dataset' not in config:
        valid_dataloader = test_dataloader
    else:
        valid_dataloader = get_dataloader(**config['valid_dataset'])

    pl_trainer = get_trainer(**config['model'])

    # 设置日志记录器
    logger = get_logger(**config['logger'])
    config['pl_trainer_func']['logger'] = logger

    # 设置断点保存回调函数
    callbacks = get_checkpoint_callpack(**config['callbacks'])
    config['pl_trainer_func']['callbacks'] = callbacks

    # 从断点处恢复
    ckpt_path = None
    if args['ckpt'] is not None:
        if os.path.exists(args['ckpt']):
            ckpt_path = args['ckpt']


    trainer = pl.Trainer(**config['pl_trainer_func'])

    # 保存配置文件
    os.makedirs(os.path.join(os.getcwd(), trainer.logger.log_dir), exist_ok=True)
    (Config.fromfile(args['config'])).dump(os.path.join(os.getcwd(), trainer.logger.log_dir, 'config.py'))

    # 开始训练
    trainer.fit(pl_trainer,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                ckpt_path=ckpt_path)


if __name__ == '__main__':
    args = parse_args()
    main(vars(args))
