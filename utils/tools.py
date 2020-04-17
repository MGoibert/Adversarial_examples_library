import logging
import os
import torch

logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.handlers = list()
torch.set_default_tensor_type(torch.DoubleTensor)

# logger definition
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    my_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    my_handler.setFormatter(formatter)
    logger.handlers = [my_handler]

    return logger

logger = get_logger("Utils")

# Device definition
nb_cuda_devices = torch.cuda.device_count()
if nb_cuda_devices > 0:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Rootpath definition
rootpath = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
