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


gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma

training_log = "training.csv"
logs_dir = "./logs"



def step_decay(epoch, base_lr, iterations_per_epoch):
    """
    Learning rate schedule - equivalent of caffe lr_policy =  "step"

    :param epoch:
    :param iterations_per_epoch:
    :return:
    """
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))
    print("Lr update: ",lrate)
    return lrate


def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Keras Pose Estimation')
    parser.add_argument('--backbone', default=None, help='backbone model name')
    parser.add_argument('--weights', default='imagenet')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--steps', default=None, type=int)
    parser.add_argument('--multiprocessing', default=False, type=bool)
    parser.add_argument('--workers', default=1, type=bool)
    parser.add_argument('--lr', default=4e-5, type=float)
    parser.add_argument('--initialepoch', default=0, type=int)
    args = parser.parse_args()


    batch_size = int(args.batchsize)

    # restore weights
    #last_epoch = restore_weights(weights_best_file, model)
    #last_epoch = restore_weights("../model/squeeze_imagenet.h5", model)



    if args.checkpoint:
        keypointnet = KeypointNet(18, args.backbone, False, None)
        model = keypointnet.model
        model.load_weights(args.checkpoint)

        # model = load_model(args.checkpoint, custom_objects={'eucl_loss': eucl_loss})
    else:
        keypointnet = KeypointNet(18, args.backbone, False, args.weights)
        model = keypointnet.model


    # print(model.summary())

    # prepare generators

    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, 'data/annotations/person_keypoints_train2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, 'data/train2017/'))
    annot_path_val = os.path.join(curr_dir, 'data/annotations/person_keypoints_val2017.json')
    img_dir_val = os.path.abspath(os.path.join(curr_dir, 'data/val2017/'))


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
                          base_lr=args.lr,
                          iterations_per_epoch=iterations_per_epoch
                          )
    lrate = LearningRateScheduler(_step_decay)

    reducelr = ReduceLROnPlateau(
        monitor='loss',
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

    callbacks_list = [lrate, checkpoint, csv_logger, tb, reducelr]


    opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # start training
    # steps_per_epoch = 5000
    if args.steps is None:
        steps_per_epoch = iterations_per_epoch
    else:
        steps_per_epoch = args.steps

    print(val_samples // batch_size)
    loss_funcs = keypointnet.keypoint_loss_function(batch_size)


    model.compile(loss=loss_funcs, optimizer=opt, metrics=["accuracy"])
    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs,
                        callbacks=callbacks_list,
                        validation_data=val_gen,
                        validation_steps=val_samples // batch_size,
                        use_multiprocessing=args.multiprocessing,
                        workers = args.workers,
                        initial_epoch=args.initialepoch)
