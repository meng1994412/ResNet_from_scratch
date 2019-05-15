# ResNet from scratch
## Objectives
Implement ResNet from scratch and train them on CIFAR-10, Tiny ImageNet, and ImageNet datasets.
* Construct ResNet56 and train the network on CIFAR-10 datasets to obtain 93.79% accuracy, which replicates the result of original ResNet on CIFAR-10.
* Use ResNet and train the network on Tiny ImageNet Visual Recognition Challenge and claim a top ranking position on Leaderboard.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.2.4 for ResNet on CIFAR-10 and 2.1.0 for the rest
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.1
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)

## Approaches
### ResNet56 on CIFAR-10
The details about CIFAR-10 datasets can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

The ResNet can be found in `resnet.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/nn/conv/resnet.py)) under `pipeline/nn/conv/` directory. The input to the model includes dimensions of the image (height, width, depth, and number of classes), number of stages, number of filters, regularization coefficient, batch normalization coefficient, batch normalization momentum, and dataset name argument (default to `cifar10`). The ResNet in this project contains a pre-activation residual module with bottleneck.

Figure 1 shows an example of the pre-activation residual module. In this project, pre-activation + bottlenect residual module is used. And Table 1 demonstrates the ResNet56 architecture for CIFAR-10. Unlike the original ResNet, which uses 7x7 filters with stride of 2 for the first convolution layer, ResNet for CIFAR-10 uses 3x3 filters with stride of 1 due to small dimensions for CIFAR-10 (32x32x3). For details about the architecture of ResNet56 for CIFAR-10, check [here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/resnet_cifar10_architecture.png).

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/preactivation_residual_module.png" width="125">

Figure 1: Pre-activation residual module ([reference](https://arxiv.org/abs/1603.05027)).

Table 1: ResNet56 for CIFAR-10

| layer name    | output size   | 56-layer                                       |
| ------------- |:-------------:| -----------------------------------------------|
| conv1         | 32 x 32 x 64  | 3 x 3, 64, stride 1                            |
| conv2_x       | 32 x 32 x 64  | [1 x 1, 16]<br>[3 x 3, 16] x 9<br>[1 x 1, 64]  |
| conv3_x       | 16 x 16 x 128 | [1 x 1, 32]<br>[3 x 3, 32] x 9<br>[1 x 1, 128] |
| conv4_x       | 8 x 8 x 256   | [1 x 1, 64]<br>[3 x 3, 64] x 9<br>[1 x 1, 256] |
| avg pool      | 1 x 1 x 256   | 8 x 8, stride 1                                |
| linear        | 256           |                                                |
| softmax       | 10            |                                                |

The `resnet_cifar10.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/resnet_cifar10.py)) is responsible for training the baseline model by using "ctrl+c" method. I can start training with an initial learning rate (and associated set of hyperparameters), monitor training, and quickly adjust the learning rate based on the results as they come in. The `TrainingMonitor` callback is responsible for plotting the loss and accuracy curves of training and validation sets. And the `EpochCheckpoint` callback is responsible for saving the model every 5 epochs.

The `resnet_cifar10_decay` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/resnet_cifar10_decay.py)) switches the method from "ctrl+c" to learning rate decay to train the network. The `TrainingMonitor` callback again is responsible for plotting the loss and accuracy curves of training and validation sets. The `LearningRateScheduler` callback is responsible for learning rate decay.

Here is the details about two callback classes:

The `trainingmonitor.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/callbacks/trainingmonitor.py)) under `pipeline/callbacks/` directory create a `TrainingMonitor` callback that will be called at the end of every epoch when training a network. The monitor will construct a plot of training loss and accuracy. Applying such callback during training will enable us to babysit the training process and spot overfitting early, allowing us to abort the experiment and continue trying to tune parameters.

The `EpochCheckpoint.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/callbacks/epochcheckpoint.py)) can help to store individual checkpoints for ResNet so that we do not have to retrain the network from beginning. The model is stored every 5 epochs.

We could use following command to train the model if we start from the beginning.
```
python resenet_cifar10.py --checkpoints output/checkpoints
```

If we start the training at middle of the epochs (simply use a number to replace `{epoch_number_you_want_to_start}`):
```
python resenet_cifar10.py --checkpoints output/checkpoints --model output/checkpoints/epoch_{epoch_number_you_want_to_start}.hdf --start_epoch {the_epoch_number_you_want_to_start}
```

For learning rate decay, just use following command:
```
python resenet_cifar10_decay.py --model output/resnet_cifar10.hdf5 --output output
```

### ResNet for Tiny ImageNet
The details about the challenge and dataset can be found [here](https://tiny-imagenet.herokuapp.com/).

The `tiny_imagenet_config.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/config/tiny_imagenet_config.py)) under `config/` directory stores all relevant configurations for the project, including the paths to input images, total number of class labels, information on the training, validation, and testing splits, path to the HDF5 datasets, and path to output models, plots, and etc.

#### Build the `HDF5` dataset
For details about how to build `HDF5` file for Tiny ImageNet dataset, check the `build_tiny_imagenet.py` in this [repo](https://github.com/meng1994412/GoogLeNet_from_scratch).

#### Build image pre-processors
The `meanpreprocessor.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/preprocessing/meanpreprocessor.py)) under `pipeline/preprocessing/` directory subtracts the mean red, green, and blue pixel intensties across the training set, which is a form of data normalization. Mean subtraction is used to reduce the affects of lighting variations during classification.

The `simplepreprocessor.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/preprocessing/simplepreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to change the size of image. This class is just used to ensure that each input image has dimenison of 64x64x3.

The `imagetoarraypreprocessor.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/preprocessing/imagetoarraypreprocessor.py)) under `pipeline/preprocessing/` directory defines a class to convert the image dataset into keras-compatile arrays.

#### Build ResNet from scratch
Table 2 demonstrates architecture of ResNet for Tiny ImageNet. ResNet for Tiny ImageNet uses 5x5 filters with stride of 1 for the first convolution layer due to small dimensions for Tiny ImageNet (64x64x3). For details about the architecture of ResNet for Tiny ImageNet, check [here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/resnet_tinyimagenet_architecture.png).

Table 2: ResNet for Tiny ImageNet.

| layer name    | output size   | 41-layer                                         |
| ------------- |:-------------:| -------------------------------------------------|
| conv1         | 64 x 64 x 64  | 5 x 5, 64, stride 1                              |
| zero padding  | 66 x 66 x 64  | 1 x 1, stride 1                                  |
| max pool      | 32 x 32 x 64  | 3 x 3, stride 2                                  |
| conv2_x       | 32 x 32 x 128 | [1 x 1, 32]<br>[3 x 3, 32] x 3<br>[1 x 1, 128]   |
| conv3_x       | 16 x 16 x 256 | [1 x 1, 64]<br>[3 x 3, 64] x 4<br>[1 x 1, 256]   |
| conv4_x       | 8 x 8 x 512   | [1 x 1, 128]<br>[3 x 3, 128] x 6<br>[1 x 1, 512] |
| avg pool      | 1 x 1 x 512   | 8 x 8, stride 1                                  |
| linear        | 512           |                                                  |
| softmax       | 200           |                                                  |

The ResNet for Tiny ImageNet can also be found in `resnet.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/nn/conv/resnet.py)). Remember to change the dataset name argument to `tiny_imagenet`.

#### Train the ResNet and evaluate it
I use a "ctrl+c" method to train the model as a baseline. By using this method, I can start training with an initial learning rate (and associated set of hyperparameters), monitor training, and quickly adjust the learning rate based on the results as they come in.

The `train.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/train.py)) is responsible for training the baseline model. The `TrainingMonitor` callback is responsible for plotting the loss and accuracy curves of training and validation sets. And the `EpochCheckpoint` callback is responsible for saving the model every 5 epochs.

After getting a sense of baseline model, I will switch to use method of learning rate decay to re-train the model. The `train_decay.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/train_decay.py)) change the method from "ctrl+c" to learning rate decay to re-train the model. The `TrainingMonitor` callback again is responsible for plotting the loss and accuracy curves of training and validation sets. The `LearningRateScheduler` callback is responsible for learning rate decay.

The `rank_accuracy.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/rank_accuracy.py)) measures the `rank-1` and `rank-5` accuracy of the model by using the testing set.

There are some helper classes for training process, including:

The `EpochCheckpoint.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/callbacks/epochcheckpoint.py)) can help to store individual checkpoints for ResNet so that we do not have to retrain the network from beginning. The model is stored every 5 epochs.

The `hdf5datasetgenerator.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/io/hdf5datasetgenerator.py)) under `pipeline/io/` directory yields batches of images and labels from `HDF5` dataset. This class can help to facilitate our ability to work with datasets that are too big to fit into memory.

The `ranked.py` ([check here](https://github.com/meng1994412/ResNet_from_scratch/blob/master/pipeline/utils/ranked.py)) under `pipeline/utils/` directory contains a helper function to measure both the `rank-1` and `rank-5` accuracy when the model is evaluated by using testing set.

We could use following command to train the model if we start from the beginning.
```
python train.py --checkpoints output/checkpoints
```

If we start the training at middle of the epochs (simply use a number to replace `{epoch_number_you_want_to_start}`):
```
python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_{epoch_number_you_want_to_start}.hdf --start_epoch {the_epoch_number_you_want_to_start}
```

For learning rate decay, just use following command:
```
python train_decay.py --model output/resnet_tinyimagenet_decay.hdf5
```

In order to use testing set to evaluate the network, use the following command:
```
python rank_accuracy.py
```

## Results
### ResNet56 on CIFAR-10
#### Experiment 1
In this experiment, I use original number of filters in ResNet for CIFAR-10, according to He et al ([reference](https://arxiv.org/abs/1512.03385)), which is (16, 32, 64) respectively for the residual module.

Figure 2 demonstrates the loss and accuracy curve of training and validation sets. And Figure 3 shows the evaluation of the network, which indicate a 88.18% accuracy. Such accuracy is quite similar to what MiniVGG obtains, according this [repo](https://github.com/meng1994412/VGGNet_from_scratch).

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet56_cifar10_1.png" width="500">

Figure 2: Plot of training and validation loss and accuracy for experiment 1.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_cifar10_experiment_1.png" width="400">

Figure 3: Evaluation of the network, indicating 88.16% accuracy, for experiment 1.

#### Experiment 2
After [experiment 1](#experiment-1), I decide to add more filters to the `conv` layers so the network can learn richer features. Thus, I change number of filters from (16, 16, 32, 64) to (64, 64, 128, 256).

Figure 4 demonstrates the loss and accuracy curve of training and validation sets for experiment 2. And Figure 5 shows the evaluation of the network, which indicate a 93.22% accuracy, for experiment 2.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet56_cifar10_2.png" width="500">

Figure 4: Plot of training and validation loss and accuracy for experiment 2.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_cifar10_experiment_2.png" width="400">

Figure 5: Evaluation of the network, indicating 93.22% accuracy, for experiment 2.

#### Experiment 3
For experiment 3, I switch method from "ctrl+c" to learning rate decay. The number of filters are still (64, 64, 128, 256). I increase the total number of epochs to 120.

Figure 6 demonstrates the loss and accuracy curve of training and validation sets for experiment 3. And Figure 7 shows the evaluation of the network, which indicate a 93.39% accuracy, for experiment 3.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet56_cifar10_3.png" width="500">

Figure 6: Plot of training and validation loss and accuracy for experiment 3.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_cifar10_experiment_3.png" width="400">

Figure 7: Evaluation of the network, indicating 93.39% accuracy, for experiment 3.

#### Experiment 4
For experiment 4, I still use the method of learning rate decay, but increase the number of epochs to 150.

Figure 8 demonstrates the loss and accuracy curve of training and validation sets for experiment 4. And Figure 9 shows the evaluation of the network, which indicate a 93.79% accuracy, for experiment 4.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet56_cifar10_5.png" width="500">

Figure 8: Plot of training and validation loss and accuracy for experiment 4.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_cifar10_experiment_5.png" width="400">

Figure 9: Evaluation of the network, indicating 93.79% accuracy, for experiment 4.

I obtain 93.79% accuracy, thus successfully replicating the work of He et al ([reference](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)) on CIFAR-10 datasets.

### ResNet on Tiny ImageNet
#### Experiment 1
For experiment 1, I use "ctrl+c" method to get a baseline model.

Figure 10 demonstrates the loss and accuracy curve of training and validation sets for experiment 1. And Figure 11 shows the evaluation of the network, indicating a 57.27% rank-1 accuracy. But clearly, the overfitting occurs, especially for epochs from 40 to 70, and it gets more severe as epoch increments.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_tinyimagenet_1.png" width="500">

Figure 10: Plot of training and validation loss and accuracy for experiment 1.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_tiny_imagenet_1.png" width="200">

Figure 11: Evaluation of the network, indicating 57.27% `rank-1` accuracy, for experiment 1.

#### Experiment 2
For experiment 2, I switch method from "ctrl+c" to learning rate decay.

Figure 12 demonstrates the loss and accuracy curve of training and validation sets for experiment 2. And Figure 13 shows the evaluation of the network, indicating a 57.93% rank-1 accuracy.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_tinyimagenet_2.png" width="500">

Figure 12: Plot of training and validation loss and accuracy for experiment 2.

<img src="https://github.com/meng1994412/ResNet_from_scratch/blob/master/output/resnet_tiny_imagenet_2.png" width="200">

Figure 13: Evaluation of the network, indicating 57.93% `rank-1` accuracy, for experiment 2.

With such `rank-1` accuracy, I can claim #5 on the Leaderboard in [Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com/).
