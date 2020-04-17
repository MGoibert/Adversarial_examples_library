import argparse
import torch
from torch import nn

from utils.tools import get_logger

torch.set_default_tensor_type(torch.DoubleTensor)

logger = get_logger("Argparser args")

def str2bool(value):
    if value in [True, "True", 'true']:
        return True
    else:
        return False

def str2list_float(values):
	val_list = [float(val) for val in values.split(";")]
	return val_list

def str2list_str(values):
    logger.info(f"values = {values}")
    if len(values) <= 1:
        val_list = [values]
    else:
        val_list = values
    return val_list


def parse_cmdline_args():
    parser = argparse.ArgumentParser(
        description="Parameters for launching the experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_name', type=str, default="MNIST",
        choices=["MNIST", "FashionMNIST", "SVHN", "SVHN_BandW", "CIFAR10"],
        help="Which dataset to use")
    parser.add_argument(
        '--architecture', type=str, default="MNIST_MLP",
        choices=["MNIST_MLP", "MNIST_LeNet", "FashionMNIST_MLP", "FashionMNIST_LeNet",
        "SVHN_LeNet", "SVHN_LeNet_BandW", "CIFAR_LeNet", "ResNet18"],
        help="Which architecture of NN to use")
    parser.add_argument(
        '--epochs', type=int, default=50,
        help="Number of epochs: number of passes to make over data")
    parser.add_argument(
        '--loss_func', default=nn.CrossEntropyLoss(),
        help="Loss function for training")
    parser.add_argument(
        '--nb_examples', type=int, default=100,
        help="Number of adversaries to save")
    parser.add_argument(
        '--attack_types', type=str2list_str, default="FGSM", nargs="+",
        help="Number of adversaries to save")
    parser.add_argument(
        '--epsilons', type=str2list_float, default="0.01;0.1;0.4",
        help='Epsilons parameters for FGSM and BIM attack. To be written in the format "eps1;eps2;eps3;eps4"...')
    parser.add_argument(
        '--num_iter', type=int, default=50,
        help='Number of iterations for BIM, DeepFool and CW attacks')
    parser.add_argument(
        '--test_size', type=int, default=100,
        help='Batch size for the test loader')
    parser.add_argument(
        '--pruning', type=float, default=0.0,
        help='Parameter for pruning. Allow pruning during training if > 0.0')
    parser.add_argument(
        '--adv_training', type=str2bool, default=False,
        help='Allow for PGD adversarial training if set to true')
    

    return parser.parse_args()