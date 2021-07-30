# -*- coding: utf-8 -*-
# @Blog    ：https://leimao.github.io/blog/PyTorch-Distributed-Training/

import numpy as np
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['NPY_MKL_FORCE_INTEL'] = '1'
os.environ['KMP_WARNINGS'] = '0'

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import time
import sys

from tensorflow.compat.v1.app import flags
from selftf.lib.mltuner.mltuner_util import MLTunerUtil
from selftf.lib.mltuner.mltuner_util import convert_model

MODELS = {
    "vgg11": torchvision.models.vgg11, # 9min/5epochs
    "squeezenet": torchvision.models.squeezenet1_0, # 1min/5epochs
    # "resnet18": torchvision.models.resnet18, 
    "mobilenet": torchvision.models.mobilenet_v2, # 3min/5epochs
    "mnasnet": torchvision.models.mnasnet1_0 # 3min/5epochs
}

EPOCHS = {
    "vgg11": 15,
    "squeezenet": 50,
    "mobilenet": 30,
    "mnasnet": 30
}

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def main():

    num_epochs_default = 5
    batch_size_default = 1024
    learning_rate_default = 0.01
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "resnet_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    # argv = parser.parse_args()
    flags.DEFINE_integer("local_rank", 0, "Local rank. Necessary for using the torch.distributed.launch utility.")
    flags.DEFINE_integer("num_epochs", num_epochs_default, "Number of training epochs.")
    flags.DEFINE_integer("batch_size", batch_size_default, "Training batch size for one process.")
    flags.DEFINE_float("learning_rate", learning_rate_default, "Learning rate")
    flags.DEFINE_integer("random_seed", random_seed_default, "Random seed.")
    flags.DEFINE_string('model_dir', model_dir_default, "Directory for saving models.")
    flags.DEFINE_string('model_filename', model_filename_default, "Model filename.")
    flags.DEFINE_string('model', "squeezenet", 'Name of the model to be run')
    
    
    mltunerUtil = MLTunerUtil()
    argv = flags.FLAGS
    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    # resume = argv.resume

    batch_size = mltunerUtil.get_batch_size()
    learning_rate = mltunerUtil.get_learning_rate()
    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Encapsulate the model on the GPU assigned to the current process
    model_name = argv.model.replace("pytorch_","")
    model = MODELS[model_name](pretrained=False)
    num_epochs = mltunerUtil.get_trial_budget()
    if num_epochs is None:
        num_epochs = EPOCHS[model_name] # default values

    if argv.get_model:
        print("Start getting model info.")
        from torch.autograd import Variable
        from pytorch2keras import pytorch_to_keras
        # save as graph
        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        input_var = Variable(torch.FloatTensor(input_np))
        # we should specify shape of the input tensor
        keras_model = pytorch_to_keras(model, input_var, [(3, 224, 224,)], verbose=False)  
        convert_model(keras_model, argv.script_path)
        return
    
    mltunerUtil.set_pytorch_hardware_param()

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend="nccl")
    torch.distributed.init_process_group(backend="gloo")

    # device = torch.device("cuda:{}".format(local_rank))
    device = torch.device("cpu")
    model = model.to(device)
    # ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    # if resume == True:
    #     map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    #     ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="selftf/tf_job/other_framework/pytorch/data", train=True, download=False, transform=transform) 
    test_set = torchvision.datasets.CIFAR10(root="selftf/tf_job/other_framework/pytorch/data", train=False, download=False, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=mltunerUtil.get_num_worker())
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=mltunerUtil.get_num_worker())

    criterion = nn.CrossEntropyLoss()
    if mltunerUtil.get_optimizer() == "Adam":
        optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)
    elif mltunerUtil.get_optimizer() == "SGD":
        optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    print("Step per epoch:", len(train_loader.dataset))
    loss = None
    st = time.time()
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
        # Save and evaluate model routinely
        # if epoch % 10 == 0:
        #     if local_rank == 0:
        #         accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
        #         # torch.save(ddp_model.state_dict(), model_filepath)
        #         print("-" * 75)
        #         print("Epoch: {}, Accuracy: {} Loss: {}".format(epoch, accuracy, loss))
        #         print("-" * 75)

        ddp_model.train()
        
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
        print("Epoch: {}, Accuracy: {} Loss: {}".format(epoch, accuracy, loss))
        sys.stderr.flush()
    mltunerUtil.report_iter_loss(epoch, loss, time.time() - st)
    print(f"Final loss:{loss}")
    

if __name__ == "__main__":
    main()