import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas
from functools import partial
import argparse
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

from network.keypoint_net import KeypointNet

from dataflow.keypoint_datagen import get_dataflow, batch_dataflow, COCODataPaths


batch_size = 4
base_lr = 1e-4
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 60000 # 600000
steps_per_epoch = 3000

weights_best_file = "weights.best.h5"
training_log = "training.csv"
logs_dir = "./logs"



def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """
    def _eucl_loss(x, y):
        print(x.shape, y.shape)
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["Dfinal_2"] = _eucl_loss
    #losses["weight_masked"] = _eucl_loss
    # losses["weight_stage1_L2"] = _eucl_loss
    # losses["weight_stage2_L1"] = _eucl_loss
    # losses["weight_stage2_L2"] = _eucl_loss
    # losses["weight_stage3_L1"] = _eucl_loss
    # losses["weight_stage3_L2"] = _eucl_loss
    # losses["weight_stage4_L1"] = _eucl_loss
    # losses["weight_stage4_L2"] = _eucl_loss
    # losses["weight_stage5_L1"] = _eucl_loss
    # losses["weight_stage5_L2"] = _eucl_loss
    # losses["weight_stage6_L1"] = _eucl_loss
    # losses["weight_stage6_L2"] = _eucl_loss

    return losses

def eucl_loss(x, y):
    l = K.sum(K.square(x - y)) / batch_size / 2
    return l


def step_decay(epoch, iterations_per_epoch):
    """
    Learning rate schedule - equivalent of caffe lr_policy =  "step"

    :param epoch:
    :param iterations_per_epoch:
    :return:
    """
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

    return lrate


def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i

def get_imagenet_weights():
    """Downloads ImageNet trained weights from Keras.
    Returns path to weights file.
    """
    from keras.utils.data_utils import get_file
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                             'releases/download/v0.2/' \
                             'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    return weights_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Keras Pose Estimation')
    parser.add_argument('--backbone', default=None, help='backbone model name')
    parser.add_argument('--ambient', default='desktop', help='local training')
    parser.add_argument('--weights', default='imagenet')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--multiprocessing', default=False, type=bool)
    parser.add_argument('--workers', default=1, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    args = parser.parse_args()


    batch_size = int(args.batchsize)

    # restore weights
    #last_epoch = restore_weights(weights_best_file, model)
    #last_epoch = restore_weights("../model/squeeze_imagenet.h5", model)

    if args.checkpoint:
        #model = KeypointNet(18, args.backbone, False, None).model
        #model.load_weights(args.checkpoint)
        model = load_model(args.checkpoint, custom_objects={'eucl_loss': eucl_loss})
    else:
        model = KeypointNet(18, args.backbone, False, args.weights).model

    print(model.summary())

    # prepare generators

    curr_dir = os.path.dirname(__file__)
    if args.ambient == "colab":
        annot_path = os.path.join(curr_dir, 'data/annotations/person_keypoints_train2017.json')
        img_dir = os.path.abspath(os.path.join(curr_dir, 'data/train2017/'))
        annot_path_val = os.path.join(curr_dir, 'data/annotations/person_keypoints_val2017.json')
        img_dir_val = os.path.abspath(os.path.join(curr_dir, 'data/val2017/'))
    else:
        annot_path = ('/home/igor/Pesquisa/Datasets/COCO/annotations/person_keypoints_val2017.json')
        img_dir = ('/home/igor/Pesquisa/Datasets/COCO/images/val2017/')
        annot_path_val = ('/home/igor/Pesquisa/Datasets/COCO/annotations/person_keypoints_val2017.json')
        img_dir_val = ('/home/igor/Pesquisa/Datasets/COCO/images/val2017/')

    # get dataflow of samples
    coco_data_train = COCODataPaths(
        annot_path=annot_path,
        img_dir=img_dir
    )

    coco_data_val = COCODataPaths(
        annot_path=annot_path_val,
        img_dir=img_dir_val
    )

    df = get_dataflow([coco_data_train])
    train_samples = df.size()

    df_val = get_dataflow([coco_data_val])
    val_samples = df_val.size()

    # get generator of batches

    batch_df = batch_dataflow(df, batch_size)
    train_gen = gen(batch_df)

    batch_df_val = batch_dataflow(df_val, batch_size)
    val_gen = gen(batch_df_val)

    # setup lr multipliers for conv layers

    #lr_multipliers = get_lr_multipliers(model)

    # configure callbacks

    iterations_per_epoch = train_samples // batch_size
    _step_decay = partial(step_decay,
                          iterations_per_epoch=iterations_per_epoch
                          )
    lrate = LearningRateScheduler(_step_decay)

    reducelr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0
    )
    checkpoint = ModelCheckpoint("model.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',
                                 verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='min', period=1)
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                     write_images=False)

    callbacks_list = [checkpoint, csv_logger, tb, reducelr]


    opt = Adam(lr=args.lr)
    # start training
    # steps_per_epoch = 5000
    print(args.steps)
    print(val_samples // batch_size)
    loss_funcs = get_loss_funcs()

    if not args.checkpoint:
        model.compile(loss=eucl_loss, optimizer=opt, metrics=["accuracy"])
    model.fit_generator(train_gen,
                        steps_per_epoch=args.steps,
                        epochs=args.epochs,
                        callbacks=callbacks_list,
                        validation_data=val_gen,
                        validation_steps=val_samples // batch_size,
                        use_multiprocessing=args.multiprocessing,
                        workers = args.workers,
                        initial_epoch=0)
