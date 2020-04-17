# Adversarial examples library

This repository contains code to produce and save adversarial examples

## 1) Setup the code

The code uses PyTorch as well as some classical libraries (numpy, etc).

To start using the code, download the repository in your desired folder.
Then, start by creating you working python environement
```bash
$ python3 -m venv my_env
$ source my_env/bin/activate
```

Then, install the required packages:
```bash
$ pip install -e .
``` 

That's it, you should be ready for the experiments !

## 2) Code organization

The main file, that you will have to run to launch the experiment is `main.py` in the root of the project.

It calls several other python file containing useful function:

- In the folder "attacks", you will find all the functions related to the implementation of attacks. The file `attacks.py` just implement FGSM, BIM, DeepFool and CW attacks. The file `generate_attacks.py` contains functions to run this attacks and produce adversaries.

- In the folder "models", you will find what is needed to download datasets, create the Neural Networks you'll use, and to train your algorithm. `dataset.py` contains the class downloading and preparing the dataset. `neural_nets.py` contains the NN algoriths available for now. If you want to add more NNs, you just have to define them in this file. `resnet.py` is a Pytorch implementation of a ResNet algo. `train.py` contains the functions to load a model, train it if needed and evaluate its accuracy.

- The folder "utils" contains helpful functions to ease the use of this code.

Calling the `main.py` file may output different folders and files. Trained models will be saved in the folder "trained_models", to prevent us losing time retraining NNs. The adversarial examples generated will be saved in the folder "saved_adversaries".

## 3) Launching an experiment

### A) Experiment parameters

There are different parameters you can use to personnalize the experiment.

##### For choosing the model
- dataset_name: which dataset you want to work on (MNIST, SVHN, CIFAR10, etc)
- architecture: which NN you want to use. To take among those in `neural_nets.py` (MNIST_MLP, SVHN_LeNet, etc)
- epochs: how many epochs to use to train the model
- loss_func: which loss function to use during training and attack generation (by defaul, Cross Entropy loss)

##### For attacks
- nb_examples: how many adversarial examples you want to generate (by default 100)
- attack_types: list of all the attacks you want to use to generate the adversarial examples (FGSM, BIM, DeepFool, CW)
- epsilons: list of epsilons parameters for FGSM or BIM attack
- num_iter: number of iteration for BIM, DeepFool or CW attacks

##### Other
- pruning: if you want to evaluate/create adversarial examples on a pruned NN (NOT IMPLEMENTED YET)
- adv_training: if you want to evaluate/create adversarial examples on a PGD-adversarially trained NN (NOT IMPLEMENTED YET)
- test_size: batch size for the test set (note that DeepFool automatically have a test_size of 1 because not implemented if > 1)

### B) Experiment examples

To launch the basic experiment (using MNIST dataset and a basic MLP with only FGSM attack) just use:

```bash
$ ipython --pdb main.py
```

Then you can modify the parameters in command line.

For example, to use SVHN LeNet on all attacks with specific values of epsilons for FGSM and BIM:

```bash
$ ipython --pdb main.py -- --dataset_name SVHN --architecture SVHN_LeNet --epochs 250 --attack_types FGSM BIM DeepFool CW --epsilons 0.01;0.025;0.05;0.1;0.2
```

NOT IMPLEMENTED YET

If you want to test these attacks on a pruned NN (here on MNIST LeNet) up to 80% (meaning you remove 80% of edges):

```bash
$ ipython --pdb main.py -- --pruning 0.8  --architecture MNIST_LeNet --epochs 200 --attack_types FGSM --epsilons 0.01;0.05;0.1;0.2;0.3;0.4
```

If you want to test these attacks on an adversarially trained NN (here on MNIST LeNet):

```bash
$ ipython --pdb main.py -- --adv_training True  --architecture MNIST_LeNet --epochs 200 --attack_types FGSM --epsilons 0.01;0.05;0.1;0.2;0.3;0.4
```
