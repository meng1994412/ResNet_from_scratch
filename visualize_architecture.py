# import packages
from pipeline.nn.conv import ResNet
from keras.utils import plot_model

# initialize the ResNet and then write the network architecture
# visualization to disk
model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg = 0.0005)
plot_model(model, to_file = "resnet_cifar10_architecture.png", show_shapes = True)
