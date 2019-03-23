import keras.layers as KL
from keras.models import Model
from network.backbone import Backbone
import keras.backend as K
import keras_resnet
import keras_resnet.models
from keras.utils import get_file

class KeypointNet():

    def __init__(self, nb_keypoints, bck_arch = 'resnet50', prediction = False, bck_weights=None):
        self.nb_keypoints = nb_keypoints + 1 # K + 1(mask)
        if prediction:
            input_image = KL.Input(shape=(None, None, 3), name='inputs')
        else:
            input_image = KL.Input(shape=(480, 480, 3), name='inputs')
        # input_heat_mask = KL.Input(shape=(120,120,19), name="mask_heat_input")
        # backbone = Backbone(input_image, bck_arch, bck_weights).model
        if bck_arch == 'resnet50':
            backbone = keras_resnet.models.FPN2D50(input_image)

        if bck_arch == 'resnet101':
            backbone = keras_resnet.models.FPN2D101(input_image)



        input_graph = backbone.input
        P2,P3,P4,P5, _ = backbone.output
        self.keypoint_net(P2,P3,P4,P5)
        # self.apply_mask(self.D, input_heat_mask)

        output_loss = [self.D, self.k2, self.k3, self.k4, self.k5]

        if prediction:
            self.model = Model(inputs=[input_graph], outputs=[self.D])
        else:
            self.model = Model(inputs=[input_graph], outputs=output_loss)

        if bck_weights is not None:
            if bck_weights == 'imagenet':
                weights = self.download_weights(bck_arch)
                self.model.load_weights(weights, by_name=True, skip_mismatch=True)
            else:
                self.model.load_weights(bck_weights, by_name=True, skip_mismatch=True)


        print(self.model.summary())

    def keypoint_net(self, P2,P3,P4,P5):

        # ### FPN ####
        # P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        # P4 = KL.Add(name="fpn_p4add")([
        #     KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        #     KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        # P3 = KL.Add(name="fpn_p3add")([
        #     KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        #     KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        # P2 = KL.Add(name="fpn_p2add")([
        #     KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        #     KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        self.P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        self.P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        self.P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)

        self.P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)


        ### KEYPOINT NET ####

        # intermidiate supervision for loss

        self.k2 = KL.Conv2D(19, (1,1), strides=1, padding="valid", name='sup_loss_k2') (self.P2)
        self.k3 = KL.Conv2D(19, (1,1), strides=1, padding="valid", name='sup_loss_k3') (self.P3)
        self.k3 = KL.UpSampling2D((2, 2), name='sup_loss_k3up')(self.k3)
        self.k4 = KL.Conv2D(19, (1, 1), strides=1, padding="valid", name='sup_loss_k4')(self.P4)
        self.k4 = KL.UpSampling2D((4, 4), name='sup_loss_k4up')(self.k4)
        self.k5 = KL.Conv2D(19, (1, 1), strides=1, padding="valid", name='sup_loss_k5')(self.P5)
        self.k5 = KL.UpSampling2D((8, 8), name='sup_loss_k5up')(self.k5)


        self.D2 = KL.Conv2D(128, (3, 3), name="d2_1", padding="same") (self.P2)
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
        self.D = KL.Conv2D(self.nb_keypoints, (1, 1), padding="SAME", name="Dfinal_2")(self.D)

    def download_weights(self, bck):
        """ Downloads ImageNet weights and returns path to weights file.
                """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if bck == 'resnet50':
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif bck == 'resnet101':
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif bck == 'resnet152':
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def keypoint_loss_function(self, batch_size):
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
        losses["sup_loss_k2"] = _eucl_loss
        losses["sup_loss_k3up"] = _eucl_loss
        losses["sup_loss_k4up"] = _eucl_loss
        losses["sup_loss_k5up"] = _eucl_loss

        return losses


# KeypointNet(18)