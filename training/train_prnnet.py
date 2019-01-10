import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import keras
from pycocotools.coco import  COCO
from network.prn_network import PRN, PRN_Seperate
from dataflow.prn_data_generator import train_bbox_generator, val_bbox_generator
from dataflow.prn_data_generator  import get_anns
from evaluating.prn_evaluate import Evaluation
from keras.callbacks import ReduceLROnPlateau

import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):

        # --------------------------  General Training Options
        self.parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
        self.parser.add_argument('--number_of_epoch', type=int, default=20, help='Epoch')
        self.parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
        self.parser.add_argument('--node_count', type=int, default=1024, help='Hidden Layer Node Count')
        # --------------------------  General Training Options

        self.parser.add_argument('--exp', type=str, default='test/', help='Experiment name')

        # --------------------------
        self.parser.add_argument('--coeff', type=int, default=2, help='Coefficient of bbox size')
        self.parser.add_argument('--threshold', type=int, default=0.21, help='BBOX threshold')
        self.parser.add_argument('--window_size', type=int, default=15, help='Windows size for cropping')
        # --------------------------

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt

class My_Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save('checkpoint/'+option.exp + 'epoch_{}.h5'.format(epoch))
        print('Epoch', epoch+1, 'has been saved')
        Evaluation(self.model, option, coco_val)
        print ('Epoch', epoch+1, 'has been tested')
        return


def main(option):
    if not os.path.exists('checkpoint/'+option.exp):
        os.makedirs('checkpoint/'+option.exp)

    model = PRN_Seperate(option.coeff*28,option.coeff*18, option.node_count)

    adam_optimizer = keras.optimizers.Adam(lr=option.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
    Own_callback = My_Callback()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2)

    model.fit_generator(generator=train_bbox_generator(coco_train, option.batch_size, option.coeff*28,option.coeff*18,option.threshold),
                        steps_per_epoch=len(get_anns(coco_train)) // option.batch_size,
                        validation_data=val_bbox_generator(coco_val, option.batch_size,option.coeff*28,option.coeff*18, option.threshold),
                        validation_steps=len(coco_val.getAnnIds()) // option.batch_size,
                        epochs=option.number_of_epoch,
                        callbacks=[Own_callback, reduce_lr],
                        verbose=1,
                        initial_epoch=0)


if __name__ == "__main__":
    option = Options().parse()
    coco_train = COCO(os.path.join('data/annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join('data/annotations/person_keypoints_val2017.json'))
    # coco_train = COCO(os.path.join('/home/igor/Pesquisa/Datasets/COCO/annotations/person_keypoints_train2017.json'))
    # coco_val = COCO(os.path.join('/home/igor/Pesquisa/Datasets/COCO/annotations/person_keypoints_val2017.json'))
    main(option)
