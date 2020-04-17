import torch
import copy
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils.tools import get_logger, device

torch.set_default_tensor_type(torch.DoubleTensor)

logger = get_logger("Attacks")

def where(cond, x, y):
    cond = cond.double()
    return (cond * x) + ((1 - cond) * y)

# Base class for attacks
class _BaseAttack(object):
    def __init__(self, model, num_iter=3, lims=(0.0, 1.0)):
        self.model = model
        self.num_iter = num_iter
        self.lims=lims

    def run(data, target, epsilon):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def clamp(self, data):
        return torch.clamp(data, *self.lims)

# FGSM
class FGSM(_BaseAttack):
    def __init__(self, model, loss_func, lims=(0.0,1.0)):
        super(FGSM, self).__init__(model)
        self.lims = lims
        self.loss_func = loss_func

    def run(self, data, target, epsilon, num_classes=10, pred=None, retain_graph=True):
        if pred is None:
            pred = self.model(data)
        loss = self.loss_func(pred, target)
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        return self.clamp(data + epsilon * data.grad.data.sign())

# BIM
class BIM(_BaseAttack):
    def __init__(self, model, loss_func, num_iter=20, lims=(0.0, 1.0)):
        super(BIM, self).__init__(model, num_iter=num_iter, lims=lims)
        self.lims = lims
        self.loss_func = loss_func

    def run(self, data, target, epsilon, num_classes=10, epsilon_iter=None):
        target = target.detach()
        if epsilon_iter is None:
            epsilon_iter = 2 * epsilon / self.num_iter

        x_ori = data.data
        for _ in range(self.num_iter):
            data = Variable(data.data, requires_grad = True)

            # forward pass
            h_adv = self.model(data)
            self.model.zero_grad()
            loss = self.loss_func(h_adv, target)

            # backward pass
            loss.backward(retain_graph=True)

            # single-step of FGSM: data <-- x_adv
            x_adv = data + epsilon_iter * data.grad.sign()
            eta = torch.clamp(x_adv - x_ori, min=-epsilon, max=epsilon)
            data = self.clamp(x_ori + eta)

        return data

# Define CW attack then CW
def _to_attack_space(x, lims=(0.0, 1.0)):
    # map from [min, max] to [-1, +1]
    a = (lims[0] + lims[1]) / 2
    b = (lims[1] - lims[0]) / 2
    x = (x - a) / b

    # from [-1, +1] to approx. (-1, +1)
    x = x * 0.999999999

    # from (-1, +1) to (-inf, +inf)
    x = 1. / 2. * torch.log((1 + x) / (1 - x))
    x = torch.clamp(x,-9,9)

    return x


def _to_model_space(x, lims=(0.0, 1.0)):
    # from (-inf, +inf) to (-1, +1)
    x = (1 - torch.exp(-2 * x * 0.999)) / (1 + torch.exp(
        -2 * x * 0.999999999))

    # map from (-1, +1) to (min, max)
    a = (lims[0] + lims[1]) / 2
    b = (lims[1] - lims[0]) / 2
    x = x * b + a

    return x

# ----------


def _soft_to_logit(softmax_list):
    soft_list = torch.clamp(softmax_list, 1e-250, (1-1e-15))
    return torch.log(soft_list)

# ----------


def _fct_to_min(adv_x, reconstruct_data, target, y_pred, logits, c, confidence=0, lims=(0.0, 1.0)):
    # Logits
    #logits = _soft_to_logit(y_pred)

    # Index of original class
    if False:
        adv_x = adv_x[:1]
        reconstruct_data = reconstruct_data[:1]
        logits = logits[:1]
    #c_min = target.data.numpy()  # .item()

    c_min = target.data # sans numpy implem
    c_max = (torch.stack( [ a != a[i] for a, i in zip(y_pred, target) ] ).double()*y_pred).max(dim=-1)[1]
    i = range(len(logits))

    is_adv_loss = torch.max(logits[i, c_min] - logits[i, c_max] + confidence, torch.zeros_like(logits[i, c_min])).to(adv_x.device)

    # Perturbation size part of the objective function: corresponds to the
    # minimization of the distance between the "true" data and the adv. data.
    scale = (lims[1] - lims[0]) ** 2
    l2_dist = ((adv_x - reconstruct_data) **2).sum(1).sum(1).sum(1) / scale

    # Objective function
    tot_dist = l2_dist.to(adv_x.device) + c.to(adv_x.device) * is_adv_loss.to(adv_x.device)
    return tot_dist

# ----------


def CW_attack(data, target, model, binary_search_steps=15, num_iter=50,
    confidence=0, learning_rate=0.05, initial_c=1, lims=(0.0, 1.0)):
    
    #data = data.unsqueeze(0)
    batch_size = 1 if len(data.size()) < 4 else len(data)
    att_original = _to_attack_space(data.detach(), lims=lims)
    reconstruct_original = _to_model_space(att_original, lims=lims)

    c = torch.ones(batch_size) * initial_c
    lower_bound = np.zeros(batch_size)
    upper_bound = np.ones(batch_size) * np.inf
    best_x = data

    for binary_search_step in range(binary_search_steps):
        perturb = [ torch.zeros_like(att_original[t], requires_grad=True)
                   for t in range(batch_size)]
        optimizer_CW = [torch.optim.Adam([perturb[t]], lr=learning_rate)
                        for t in range(batch_size)]
        found_adv = torch.zeros(batch_size).byte()

        for iteration in range(num_iter):
            x = torch.clamp(_to_model_space(att_original + torch.cat([perturb_.unsqueeze(0) for perturb_ in perturb]) , lims=lims), *lims)
            y_pred = model(x)
            logits = model(x, output="presoft")
            cost = _fct_to_min(x, reconstruct_original, target, y_pred, logits, c,
                               confidence, lims=lims)

            for t in range(batch_size):
                optimizer_CW[t].zero_grad()
                cost[t].backward(retain_graph=True)
                optimizer_CW[t].step()
                if logits[t].squeeze().argmax(-1, keepdim=True).item() != target[t]:
                    #if found_adv[t] == 0: logger.info(f"!! Found adv !! at BSS = {binary_search_step} and iter = {iteration}")
                    found_adv[t] = 1
                else:
                    found_adv[t] = 0

        for t in range(batch_size):
            if found_adv[t]:
                upper_bound[t] = c[t]
                best_x[t] = x[t]
            else:
                lower_bound[t] = c[t]
            if upper_bound[t] == np.inf:
                c[t] = 10 * c[t]
            else:
                c[t] = (lower_bound[t] + upper_bound[t]) / 2

    return best_x.squeeze(0)

class CW(_BaseAttack):
    """
    Carlini-Wagner Method
    """
    def __init__(self, model, binary_search_steps=15,
                 num_iter=50, lims=(0.0, 1.0)):
        _BaseAttack.__init__(self, model, lims=lims)
        self.binary_search_steps = binary_search_steps
        self.num_iter = num_iter
        self.lims = lims

    def run(self, data, target, **kwargs):
        perturbed_data = CW_attack(
            data, target, self.model, num_iter=self.num_iter,
            binary_search_steps=self.binary_search_steps, lims=self.lims, **kwargs)
        return perturbed_data



class DeepFool(_BaseAttack):
    def __init__(self, model, num_classes,  num_iter=5):
        super(DeepFool, self).__init__(model, num_iter=num_iter)
        self.num_classes = num_classes
        self.num_iter = num_iter
        self.model = model

    def run(self, image, true_label, epsilon=None):
        self.model.eval()

        nx = torch.unsqueeze(image, 0).detach().cpu().numpy().copy()
        nx = torch.from_numpy(nx)
        nx.requires_grad = True
        eta = torch.zeros(nx.shape)

        out = self.model(nx+eta, output="presoft")
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.num_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(self.num_classes):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())


                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone() if type(ri) != type(None) else 0
            nx.grad.data.zero_()
            out = self.model(self.clamp(nx+eta), output="presoft")
            py = out.max(1)[1].item()
            i_iter += 1
        
        x_adv = self.clamp(nx+eta)
        x_adv.squeeze_(0)

        return x_adv