import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas
from functools import partial
import argparse
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.optimizers import Adam


from network.keypoint_net import KeypointNet

from dataflow.keypoint_datagen import get_dataflow, batch_dataflow, COCODataPaths


batch_size = 2
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

weights_best_file = "weights.best.h5"
training_log = "training.csv"
logs_dir = "./logs"

from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def get_last_epoch():
    """
    Retrieves last epoch from log file updated during training.

    :return: epoch number
    """
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)



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
    args = parser.parse_args()


    # restore weights
    #last_epoch = restore_weights(weights_best_file, model)
    #last_epoch = restore_weights("../model/squeeze_imagenet.h5", model)
    if args.weights == 'imagenet':
        model = KeypointNet(18, args.backbone, False, args.weights).model
    else:
        model = KeypointNet(18, 'resnet50', False, None).model
        model.load_weights(args.weights)
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
    val_samples = df.size()

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
    checkpoint = ModelCheckpoint("model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
                                 verbose=0, save_best_only=False,
                                 save_weights_only=True, mode='min', period=1)
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                     write_images=False)

    callbacks_list = [lrate, checkpoint, csv_logger, tb]


    opt = Adam(lr=1e-4)
    # start training

    loss_funcs = get_loss_funcs()
    model.compile(loss=loss_funcs, optimizer=opt, metrics=["accuracy"])
    model.fit_generator(train_gen,
                        steps_per_epoch=5000,
                        epochs=max_iter,
                        callbacks=callbacks_list,
                        validation_data=val_gen,
                        validation_steps=val_samples // batch_size,
                        use_multiprocessing=False,
                        initial_epoch=0)
