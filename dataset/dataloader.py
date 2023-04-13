import dataset
import torch
import torchvision.transforms as transforms


def get_dataloader(*args, **kwargs):
    """Get dataloader 

    Returns:
        torch.utils.data.DataLoader: train or test DataLoader
    """
    
    transform_list = [
        getattr(transforms, k)() if v is None else getattr(transforms, k)(**v)
        for k, v in kwargs['transform'].items()
    ]
    kwargs['dataset_arg']['transform'] = transforms.Compose(transform_list)

    se_dataset = getattr(dataset, kwargs['name'])(**kwargs['dataset_arg'])

    se_dataloader = torch.utils.data.DataLoader(dataset=se_dataset,
                                                **kwargs['dataloader_arg'])
    
    return se_dataloader