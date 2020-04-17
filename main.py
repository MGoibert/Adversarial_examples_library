import argparse
import torch
from torch import nn
import os
import pickle

from utils.tools import get_logger, device, rootpath
from utils.argparse_args import parse_cmdline_args
from attacks.generate_attacks import gererating_adv_dict
from models.train import get_my_model
from models.datasets import Dataset
torch.set_default_tensor_type(torch.DoubleTensor)

logger = get_logger("Main")

# Parameters
args = parse_cmdline_args()
#dataset_name = "MNIST"
#architecture = "MNIST_MLP"
#epochs = 3
#nb_examples = 30
loss_func = nn.CrossEntropyLoss()
#attack_types = ["FGSM", "BIM", "DeepFool", "CW"]
#epsilons = [0.025, 0.05, 0.1, 0.4]
#num_iter = 50
#pruning = 0
#adv_training = False
#test_size = 5
logger.info(f"attack types = {args.attack_types}")


# Main function
def run_exp(nb_examples, dataset_name, model, attack_types, epsilons, num_iter, test_size):

    list_adv_dict = list()
    for attack_type in attack_types:
        logger.info(f"The current attack = {attack_type}")
        if attack_type == "DeepFool":
            test_size = 1

        # Get dataset
        dataset = Dataset.get_or_create(name=dataset_name, test_size=test_size)

        # Generate adversarial examples
        list_adv_dict.append(gererating_adv_dict(attack_type, model, dataset.test_loader,
            epsilons, nb_examples, num_iter=num_iter))

    if len(attack_types) > 1:
        final_adv_dict = {**list_adv_dict[0], **list_adv_dict[1]}
        if len(attack_types) > 2:
            for k in range(len(attack_types)-2):
                final_adv_dict = {**final_adv_dict, **list_adv_dict[2+k]}
    else:
        final_adv_dict = list_adv_dict[0]

    return final_adv_dict


# Save
def save_dict_adv(obj, pathname):
    path = pathname + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved file at: {path}")
    
def load_obj(pathname):
    path = pathname + ".pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


# Running experiments
model, filename = get_my_model(args.dataset_name, args.architecture, args.epochs,
    loss_func, args.pruning, args.adv_training)
adv_dict = run_exp(args.nb_examples, args.dataset_name, model, args.attack_types,
    args.epsilons, args.num_iter, args.test_size)

save_path = f"{rootpath}/saved_adversaries/{args.dataset_name}/{args.architecture}"
if not os.path.exists(save_path):
        os.makedirs(save_path)

if args.pruning > 0.0:
    pruning_message = f"_{args.pruning}_pruning"
else:
    pruning_message = ""
if args.adv_training:
    adv_training_message = f"_pgd_adv_train"
else:
    adv_training_message = f""

save_name = (f"{args.epochs}_epochs"
    + f"{pruning_message}"
    + f"{adv_training_message}"
    + f"_{args.attack_types}"
    + f"_{args.nb_examples}_examples"
    )

save_dict_adv(adv_dict, save_path+"/"+save_name )

logger.info(f"adv dict keys = {adv_dict.keys()}")
for at in adv_dict.keys():
    if isinstance(list(adv_dict[at].keys())[0], float):
        for eps in adv_dict[at].keys():
            logger.info(f"Attack {at}, eps = {eps} and count = {len(adv_dict[at][eps]['y'])}")
    else:
        logger.info(f"Attack {at}, and count = {len(adv_dict[at]['y'])}")




