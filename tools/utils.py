from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def get_logger(*args, **kwargs):
    logger = []
    for k, v in kwargs.items():
        logger.append(
            getattr(loggers, v['name'])(**v['logger_arg'])
        )

    return logger

def get_checkpoint_callpack(*args, **kwargs):
    callbacks = []
    for k, v in kwargs.items():
        callbacks.append(
            ModelCheckpoint(**v)
        )

    return callbacks

