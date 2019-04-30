import keras.layers as KL
from keras.models import Model
from network.backbone import Backbone
from utils.prn_forward_preprocess import get_prn_input, get_prn_output_shape
import numpy as np
import cv2

from network.retinanet import *
from network.prn_network import *
import keras.backend as K
import keras_resnet.models

class PoseCNet():

    def __init__(self, bck_arch="resnet50", nb_keypoints = 18, prn=False):
        self.nb_keypoints = nb_keypoints +1  # K + 1(mask)
        input_image = KL.Input(shape=(None, None, 3), name='inputs')
        height = 56
        width = 36
        node_count = 1024
        self.prn = prn

        #Backbone (resnet50/101/x50/x101)
        if bck_arch == 'resnet50':
            backbone = keras_resnet.models.ResNet2D50(input_image, include_top=False, freeze_bn=True)

        if bck_arch == 'resnet101':
            backbone = keras_resnet.models.ResNet2D101(input_image, include_top=False, freeze_bn=True)

        input_graph = backbone.input
        c2,c3,c4,c5 = backbone.output

        pyramid_5 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c5_reduced"
        )(c5)

        upsampled_p5 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p5_upsampled",
            size=(2, 2)
        )(pyramid_5)

        pyramid_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c4_reduced"
        )(c4)

        pyramid_4 = keras.layers.Add(
            name="p4_merged"
        )([upsampled_p5, pyramid_4])

        upsampled_p4 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p4_upsampled",
            size=(2, 2)
        )(pyramid_4)

        pyramid_4 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p4"
        )(pyramid_4)

        pyramid_3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c3_reduced"
        )(c3)

        pyramid_3 = keras.layers.Add(
            name="p3_merged"
        )([upsampled_p4, pyramid_3])

        upsampled_p3 = keras.layers.UpSampling2D(
            interpolation="bilinear",
            name="p3_upsampled",
            size=(2, 2)
        )(pyramid_3)

        pyramid_3 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p3"
        )(pyramid_3)

        pyramid_2 = keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            name="c2_reduced"
        )(c2)

        pyramid_2 = keras.layers.Add(
            name="p2_merged"
        )([upsampled_p3, pyramid_2])

        pyramid_2 = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            name="p2"
        )(pyramid_2)


        self.D2 = KL.Conv2D(128, (3, 3), name="d2_1", padding="same")(pyramid_2)
        self.D2 = KL.Conv2D(128, (3, 3), name="d2_1_2", padding="same")(self.D2)
        self.D3 = KL.Conv2D(128, (3, 3), name="d3_1", padding="same")(pyramid_3)
        self.D3 = KL.Conv2D(128, (3, 3), name="d3_1_2", padding="same")(self.D3)
        self.D4 = KL.Conv2D(128, (3, 3), name="d4_1", padding="same")(pyramid_4)
        self.D4 = KL.Conv2D(128, (3, 3), name="d4_1_2", padding="same")(self.D4)
        self.D5 = KL.Conv2D(128, (3, 3), name="d5_1", padding="same")(pyramid_5)
        self.D5 = KL.Conv2D(128, (3, 3), name="d5_1_2", padding="same")(self.D5)

        self.D3 = KL.UpSampling2D((2, 2), name='d3_up')(self.D3)
        self.D4 = KL.UpSampling2D((4, 4), name='d4_up')(self.D4)
        self.D5 = KL.UpSampling2D((8, 8), name='d5_up')(self.D5)

        self.concat = KL.concatenate([self.D2, self.D3, self.D4, self.D5], axis=-1)
        self.D = KL.Conv2D(512, (3, 3), activation="relu", padding="SAME", name="Dfinal_1")(self.concat)
        self.keypoint_output = KL.Conv2D(self.nb_keypoints, (1, 1), padding="SAME", name="Dfinal_2")(self.D)


        #DetectionNet part (RetinaNet)

        retina_net = retinanet(input_graph, [c3,c4,c5], 1)
        retina_bbox = retinanet_bbox(retina_net)
        self.detection_output = retina_bbox.output

        output = [self.keypoint_output]
        output.extend(self.detection_output)

        if self.prn:
            lambda_input = [self.keypoint_output]
            lambda_input.extend(self.detection_output)

            lambda_layer = keras.layers.Lambda(get_prn_input, get_prn_output_shape)
            self.person_heatmap = lambda_layer(lambda_input)
            #
            # #PRN NETWORK
            input = keras.layers.Lambda(lambda x: x[:,:,:,:18], name="prn_lambda_input")(self.person_heatmap)

            y = Flatten(name='prn_flatten', batch_input_shape=(None,56,36,18))(input)
            x = Dense(node_count, activation='relu', name="prn_dense_1")(y)
            x = Dropout(0.5, name='do_1')(x)
            x = Dense(width * height * 18, activation='relu', name='prn_dense_2')(x)
            x = keras.layers.Add(name="prn_dense1_add_dense2")([x, y])
            x = keras.layers.Activation('softmax', name='prn_activation')(x)
            self.prn_output = Reshape((height, width, 18), name="prn_reshape")(x)
            self.prn_output = keras.layers.Lambda(lambda x: K.tf.expand_dims(x, axis=0))(self.prn_output)

            output.append(self.prn_output)

        self.model = Model(inputs=input_graph, outputs=output)
        print(self.model.summary())

    def load_subnet_weights(self, k_weights, d_weights, p_weights):
        self.model.load_weights(k_weights, by_name=True)
        self.model.load_weights(d_weights, by_name=True)

        if self.prn:
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