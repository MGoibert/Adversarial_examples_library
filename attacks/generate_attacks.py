import torch
from torch import nn
import numpy as np

from utils.tools import get_logger, device
from attacks.attacks import FGSM, BIM, CW, DeepFool

torch.set_default_tensor_type(torch.DoubleTensor)

logger = get_logger("Generate Attacks")

def adv_gen(model, x, y,
            epsilon=0.25,
            loss_func=nn.CrossEntropyLoss(),
            num_classes=10,
            attack_type='FGSM',
            num_iter=50,
            lims=(0.0,1.0)):
    #y = torch.from_numpy(y).double()
    x = x.double()
    x, y, model = x.to(device), y.to(device), model.to(device)
    x.requires_grad = True
    if attack_type == "FGSM":
        attacker = FGSM(model, loss_func, lims=lims)
    elif attack_type == "BIM":
        attacker = BIM(model, loss_func, lims=lims, num_iter=num_iter)
    elif attack_type == "DeepFool":
        attacker = DeepFool(model, num_classes=num_classes, num_iter=num_iter)
    elif attack_type == "CW":
        attacker = CW(model, lims=lims, num_iter=num_iter)
    else:
        raise NotImplementedError(attack_type)

    if attack_type in ["FGSM", "BIM"]:
        x_adv = attacker(x, y, epsilon)
    elif attack_type == "CW":
        x_adv = attacker(x, y)
    elif attack_type == "DeepFool":
        x_adv = attacker(x, y)

    x_adv = x_adv.to(device)

    return x_adv, x

def gererating_adv_dict(attack_type, model, test_loader, epsilons, nb_examples,
    num_classes=10, num_iter=50, lims=(0.0,1.0)):
    adv_dict = dict()

    #for attack_type in attack_types:

    if attack_type in ["FGSM", "BIM"]:
        adv_dict[attack_type] = dict()
        for epsilon in epsilons:
            count = 0
            success_adv = 0
            adv_dict[attack_type][epsilon] = {k: list() for k in ["x", "y", "x_adv", "y_adv"]}
            for x, y in test_loader:
                x, y = x.detach().to(device), y.detach().to(device)
                x = x.double()
                y_pred = model(x)
                pred = y_pred.argmax(1)
                ok_mask = pred == y
                if ok_mask.sum() == 0:
                    continue

                #count += ok_mask.sum()
                x = x[ok_mask,:,:,:]
                y = y[ok_mask]

                if epsilon == 0.0:
                    for x_, y_, pred_ in zip(x, y, pred):
                        if len(adv_dict[attack_type][epsilon]["y"]) < nb_examples:
                            count += 1
                            adv_dict[attack_type][epsilon]["x"].append(x_)
                            adv_dict[attack_type][epsilon]["y"].append(y_)
                            adv_dict[attack_type][epsilon]["x_adv"].append(x_)
                            adv_dict[attack_type][epsilon]["y_adv"].append(pred_)
                else:
                    x_adv, _ = adv_gen(model, x, y, attack_type=attack_type, epsilon=epsilon,
                        num_classes=num_classes, num_iter=num_iter, lims=lims)
                    x_adv = x_adv.detach().to(device)
                    y_pred_adv = model(x_adv)

                    for x_, y_, x_adv_, y_pred_adv_ in zip(x, y, x_adv, y_pred_adv):
                        count += 1
                        if y_pred_adv_.argmax(0) != y_:
                            success_adv += 1
                            logger.info(f"x_adv type = {x_adv_.size()}")
                            adv_dict[attack_type][epsilon]["x"].append(x_)
                            adv_dict[attack_type][epsilon]["y"].append(y_)
                            adv_dict[attack_type][epsilon]["x_adv"].append(x_adv_)
                            adv_dict[attack_type][epsilon]["y_adv"].append(y_pred_adv_.argmax(0))
                        if len(adv_dict[attack_type][epsilon]["y"]) >= nb_examples:
                            break
                if len(adv_dict[attack_type][epsilon]["y"]) >= nb_examples:
                    break

            logger.info(f"Attack {attack_type} for eps = {epsilon}: adv acc = {np.round(1.0-float(success_adv)/float(count), 3)}")

    else:
        count = 0
        success_adv = 0
        adv_dict[attack_type] = {k: list() for k in ["x", "y", "x_adv", "y_adv"]}
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.double()
            y_pred = model(x)
            pred = y_pred.argmax(1)
            ok_mask = pred == y
            if ok_mask.sum() == 0:
                continue

            x = x[ok_mask,:,:,:]
            y = y[ok_mask]
            #count += ok_mask.sum()

            x_adv, _ = adv_gen(model, x, y, attack_type=attack_type,
                    num_classes=num_classes, num_iter=num_iter, lims=lims)
            y_pred_adv = model(x_adv)

            for x_, y_, x_adv_, y_pred_adv_ in zip(x, y, x_adv, y_pred_adv):
                count += 1
                if y_pred_adv_.argmax(0) != y_:
                    success_adv += 1
                    adv_dict[attack_type]["x"].append(x_)
                    adv_dict[attack_type]["y"].append(y_)
                    adv_dict[attack_type]["x_adv"].append(x_adv)
                    adv_dict[attack_type]["y_adv"].append(y_pred_adv_.argmax(0))
                if len(adv_dict[attack_type]["y"]) >= nb_examples:
                    break
            if len(adv_dict[attack_type]["y"]) >= nb_examples:
                    break
        logger.info(f"Attack {attack_type}: adv acc = {np.round(1.0-float(success_adv)/float(count), 3)}")

    return adv_dict
