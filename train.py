# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import packages
from config import tiny_imagenet_config as config
from pipeline.preprocessing import ImageToArrayPreprocessor
from pipeline.preprocessing import SimplePreprocessor
from pipeline.preprocessing import MeanPreprocessor
from pipeline.callbacks import EpochCheckpoint
from pipeline.callbacks import TrainingMonitor
from pipeline.io import HDF5DatasetGenerator
from pipeline.nn.conv import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
import argparse
import json

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required = True,
    help = "path to output checkpoint directory")
ap.add_argument("-m", "--model", type = str,
    help = "path to specific model checkpoint to load")
ap.add_argument("-s", "--start_epoch", type = int, default = 0,
    help = "epoch to restart training at")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 18, zoom_range = 0.15,
    width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15,
    horizontal_flip = True, fill_mode = "nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug = aug,
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = ResNet.build(64, 64, 3, config.NUM_CLASSES, (3, 4, 6),
        (64, 128, 256, 512), reg = 0.0005, dataset = "tiny_imagenet")
    opt = SGD(lr = 1e-1, momentum = 0.9)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

callbacks = [
    EpochCheckpoint(args["checkpoints"], every = 5, startAt = args["start_epoch"]),
    TrainingMonitor(config.FIG_PATH, jsonPath = config.JSON_PATH, startAt = args["start_epoch"])
]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 64,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages // 64,
    epochs = 10,
    max_queue_size = 10,
    callbacks = callbacks,
    verbose = 1
)

# close the databases
trainGen.close()
valGen.close()
