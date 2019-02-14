import keras.layers as KL
from keras.models import Model
from network.backbone import Backbone
from utils.prn_forward_preprocess import get_prn_input, get_prn_output_shape
import sys
import os
from scipy import ndimage
import numpy as np
import cv2

from network.retinanet import *
from network.prn_network import *
import keras.backend as K


class PoseCNet():

    def __init__(self, bck_arch="resnet50", nb_keypoints = 18):
        self.nb_keypoints = nb_keypoints + 1  # K + 1(mask)
        input_image = (None,None,3)
        height = 56
        width = 36
        node_count = 1024

        #Backbone (resnet50/101/x50/x101)
        self.backbone = Backbone(input_image, bck_arch)
        C2, C3, C4, C5 = self.backbone.model.output

        #Keypoint net part
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        self.P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        self.P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        self.P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)

        self.P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)

        self.D2 = KL.Conv2D(128, (3, 3), name="d2_1", padding="same")(self.P2)
        self.D2 = KL.Conv2D(128, (3, 3), name="d2_1_2", padding="same")(self.D2)
        self.D3 = KL.Conv2D(128, (3, 3), name="d3_1", padding="same")(self.P3)
        self.D3 = KL.Conv2D(128, (3, 3), name="d3_1_2", padding="same")(self.D3)
        self.D3 = KL.UpSampling2D((2, 2), )(self.D3)
        self.D4 = KL.Conv2D(128, (3, 3), name="d4_1", padding="same")(self.P4)
        self.D4 = KL.Conv2D(128, (3, 3), name="d4_1_2", padding="same")(self.D4)
        self.D4 = KL.UpSampling2D((4, 4))(self.D4)
        self.D5 = KL.Conv2D(128, (3, 3), name="d5_1", padding="same")(self.P5)
        self.D5 = KL.Conv2D(128, (3, 3), name="d5_1_2", padding="same")(self.D5)
        self.D5 = KL.UpSampling2D((8, 8))(self.D5)

        self.concat = KL.concatenate([self.D2, self.D3, self.D4, self.D5], axis=-1)
        self.D = KL.Conv2D(512, (3, 3), activation="relu", padding="SAME", name="Dfinal_1")(self.concat)
        self.keypoint_output = KL.Conv2D(self.nb_keypoints, (1, 1), padding="SAME", name="Dfinal_2")(self.D)


        #DetectionNet part (RetinaNet)

        retina_net = retinanet(self.backbone.model.input, [C3, C4, C5], 1)
        retina_bbox = retinanet_bbox(retina_net)
        self.detection_output = retina_bbox.output

        lambda_input = [self.keypoint_output]
        lambda_input.extend(self.detection_output)

        lambda_layer = keras.layers.Lambda(get_prn_input, get_prn_output_shape)
        self.person_heatmap = lambda_layer(lambda_input)

        #PRN NETWORK
        input = keras.layers.Lambda(lambda x: x[:,:,:,:18], name="prn_lambda_input")(self.person_heatmap)
        y = Flatten(name="prn_flatten", batch_input_shape=(None,56,36,18))(input)
        x = Dense(node_count, activation='relu', name="prn_dense_1")(y)
        x = Dropout(0.5)(x)
        x = Dense(width * height * 18, activation='relu', name="prn_dense_2")(x)
        x = keras.layers.Add(name="prn_dense1_add_dense2")([x, y])
        out = []
        start = 0
        end = width * height

        for i in range(18):
            o = keras.layers.Lambda(lambda x: x[:, start:end], name="prn_lambda_" + str(i))(x)
            o = Activation('softmax', name="prn_activation_" + str(i))(o)
            out.append(o)
            start = end
            end = start + width * height

        x = keras.layers.Concatenate(name="prn_concat")(out)
        self.prn_output = Reshape((height, width, 18), name="prn_reshape")(x)
        self.prn_output = keras.layers.Lambda(lambda x: K.tf.expand_dims(x, axis=0))(self.prn_output)


        output = [self.keypoint_output]
        output.extend(self.detection_output)
        output.append(self.prn_output)
        self.model = Model(inputs=self.backbone.model.input, outputs=output)
        print(self.model.summary())

    def load_subnet_weights(self, k_weights, d_weights, p_weights):
        self.model.load_weights(k_weights, by_name=True)
        self.model.load_weights(d_weights, by_name=True)
        self.model.load_weights(p_weights, by_name=True)


    def predict(self,image):
        shape_dst = np.max(image.shape)
        self.scale = float(shape_dst) / 480
        pad_size = np.abs(image.shape[1] - image.shape[0])
        img_resized = np.pad(image, ([0, pad_size], [0, pad_size], [0, 0]), 'constant')[:shape_dst, :shape_dst]
        self.img_input = cv2.resize(img_resized, (480, 480))
        prediction = self.model.predict(np.expand_dims(self.img_input, 0))

        return prediction

# PoseCNet(bck_arch='resnet50')