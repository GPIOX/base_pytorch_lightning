import model

def get_model(*args, **kwargs):
    pass

def get_trainer(*args, **kwargs):

    return getattr(
        model, kwargs['trainer']['name'])(args, **kwargs)
