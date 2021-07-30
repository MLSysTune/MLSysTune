import argparse
import logging
import os
import zipfile
import time

import mxnet as mx
import horovod.mxnet as hvd
from mxnet import autograd, gluon, nd
from mxnet.test_utils import download

from tensorflow.compat.v1.app import flags
from selftf.lib.mltuner.mltuner_util import MLTunerUtil
from selftf.lib.mltuner.mltuner_util import convert_model

import numpy as np

from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
from gluoncv.model_zoo import get_model as get_mxnet_model
import random


class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with index `part_index`
    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len

# model reference: https://cv.gluon.ai/model_zoo/classification.html#cifar10
MODELS = {
    "resnet20": "CIFAR_ResNet20_v2", # 0.5min/epoch
    # "resnet56": "CIFAR_ResNet56_v2",
    "wideresnet16": "CIFAR_WideResNet16_10", # 9min/epoch
    "wideresnet28": "CIFAR_WideResNet28_10", # 19min/epoch
}

EPOCHS = {
    "resnet20": 20,
    "wideresnet16": 3,
    "wideresnet28": 1
}

# Function to evaluate accuracy for a model
def evaluate(model, data_iter, context):
    metric = mx.metric.Accuracy()
    for _, batch in enumerate(data_iter):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=[context], batch_axis=0)
        label = mx.gluon.utils.split_and_load(batch[1], ctx_list=[context], batch_axis=0)
        outputs = [model(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

def main():
    flags.DEFINE_integer("batch_size", 64, "Training batch size for one process.")
    flags.DEFINE_string('dtype', 'float32', 'training data type')
    flags.DEFINE_integer("epochs", 1, "number of training epochs")
    flags.DEFINE_float("lr", 0.01, "Learning rate")
    flags.DEFINE_float("momentum", 0.9, "SGD momentum")
    flags.DEFINE_bool('no_cuda', True, 'disable training on GPU')
    flags.DEFINE_float("gradient_predivide_factor", 1.0, "apply gradient predivide factor in optimizer")
    flags.DEFINE_string('model', "resnet20", 'Name of the model to be run')

    mltunerUtil = MLTunerUtil()
    args = flags.FLAGS
    batch_size = mltunerUtil.get_batch_size()
    learning_rate = mltunerUtil.get_learning_rate()

    if not args.no_cuda:
        # Disable CUDA if there are no GPUs.
        if not mx.test_utils.list_gpus():
            args.no_cuda = True

    logging.basicConfig(level=logging.INFO)

    # Initialize Horovod
    hvd.init()

    # Horovod: pin context to local rank
    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
    num_workers = hvd.size()
    print(f"num_workers:{num_workers}")

    # Load training and validation data
    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    train_data = gluon.data.DataLoader(
        mx.gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=False, last_batch='discard', 
        sampler=SplitSampler(50000, num_workers, hvd.rank()))
    print(f"len(train_data):{len(train_data)}")
    val_data = gluon.data.DataLoader(
        mx.gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False)

    # Build model
    model_name = args.model.replace("mxnet_","")
    model = get_mxnet_model(MODELS[model_name])
    num_epochs = EPOCHS[model_name]
    model.cast(args.dtype)
    model.hybridize()

    # Create optimizer
    optimizer_params = {'momentum': args.momentum,
                        'learning_rate': learning_rate * hvd.size()}
    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Initialize parameters
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                magnitude=2)
    model.initialize(initializer, ctx=context)

    if args.get_model:
        print("Start getting model info.")
        from gluon2pytorch import gluon2pytorch
        from torch import FloatTensor
        from torch.autograd import Variable
        from pytorch2keras import pytorch_to_keras

        pytorch_model = gluon2pytorch(model, [(1,3,28,28)], dst_dir=None, pytorch_module_name='intermediate_pytorch_model')
        input_np = np.random.uniform(0, 1, (1, 3, 28,28))
        input_var = Variable(FloatTensor(input_np))
        # we should specify shape of the input tensor
        keras_model = pytorch_to_keras(pytorch_model, input_var, [(3, 28,28,)], verbose=False)  
        convert_model(keras_model, args.script_path)
        return

    mltunerUtil.set_mxnet_hardware_param()
    # Horovod: fetch and broadcast parameters
    params = model.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt,
                                    gradient_predivide_factor=args.gradient_predivide_factor)

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()
    st = time.time()

    # Train model
    for epoch in range(num_epochs):
        train_loss = 0
        tic = time.time()
        metric.reset()
        for nbatch, batch in enumerate(train_data, start=1):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[context], batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=[context], batch_axis=0)
            with autograd.record():
                output = [model(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(args.batch_size)
            metric.update(label, output)
            train_loss += sum([l.sum().asscalar() for l in loss])

            if nbatch % 10 == 0:
                name, acc = metric.get()
                logging.info('[Epoch %d Batch %d] Training: %s=%f' %
                            (epoch, nbatch, name, acc))

        train_loss /= args.batch_size * len(train_data)
        if hvd.rank() == 0:
            elapsed = time.time() - tic
            speed = nbatch * args.batch_size * hvd.size() / elapsed
            logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
                        epoch, speed, elapsed)

        # Evaluate model accuracy
        _, train_acc = metric.get()
        name, val_acc = evaluate(model, val_data, context)
        if hvd.rank() == 0:
            logging.info('Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f', epoch, name,
                        train_acc, name, val_acc)

    mltunerUtil.report_iter_loss(args.epochs, train_loss, time.time() - st)

if __name__ == "__main__":
    main()