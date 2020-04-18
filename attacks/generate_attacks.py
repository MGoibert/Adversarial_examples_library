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
            adv_dict[attack_type][epsilon] = {k: list() for k in ["x", "y", "x_adv", "y_adv"]}
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
                count += ok_mask.sum()

                x_adv, _ = adv_gen(model, x, y, attack_type=attack_type, epsilon=epsilon,
                    num_classes=num_classes, num_iter=num_iter, lims=lims)
                y_pred_adv = model(x_adv)

                for x_, y_, x_adv_, y_pred_adv_ in zip(x, y, x_adv, y_pred_adv):
                    if y_pred_adv_.argmax(0) != y_:
                        adv_dict[attack_type][epsilon]["x"]+= list(x_)
                        adv_dict[attack_type][epsilon]["y"]+= [y_]
                        adv_dict[attack_type][epsilon]["x_adv"] += list(x_adv_)
                        adv_dict[attack_type][epsilon]["y_adv"]+= [y_pred_adv_.argmax(0)]
                    if len(adv_dict[attack_type][epsilon]["y"]) >= nb_examples:
                        break
                if len(adv_dict[attack_type][epsilon]["y"]) >= nb_examples:
                        break

            logger.info(f"Attack {attack_type} for eps = {epsilon}: adv acc = {np.round(1.0-float(len(adv_dict[attack_type][epsilon]['y']))/float(count), 3)}")

    else:
        count = 0
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
            count += ok_mask.sum()

            x_adv, _ = adv_gen(model, x, y, attack_type=attack_type,
                    num_classes=num_classes, num_iter=num_iter, lims=lims)
            y_pred_adv = model(x_adv)

            for x_, y_, x_adv_, y_pred_adv_ in zip(x, y, x_adv, y_pred_adv):
                if y_pred_adv_.argmax(0) != y_:
                    adv_dict[attack_type]["x"] += list(x_)
                    adv_dict[attack_type]["y"] += [y_]
                    adv_dict[attack_type]["x_adv"] += list(x_adv)
                    adv_dict[attack_type]["y_adv"] += [y_pred_adv_.argmax(0)]
                if len(adv_dict[attack_type]["y"]) >= nb_examples:
                    break
            if len(adv_dict[attack_type]["y"]) >= nb_examples:
                    break
        logger.info(f"Attack {attack_type}: adv acc = {np.round(1.0-float(len(adv_dict[attack_type]['y']))/float(count), 3)}")

    return adv_dict
