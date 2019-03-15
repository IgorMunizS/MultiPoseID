import keras.layers as KL
from keras.models import Model
from network.backbone import Backbone


class KeypointNet():

    def __init__(self, nb_keypoints, bck_arch = 'resnet50', prediction = False, bck_weights=None):
        self.nb_keypoints = nb_keypoints + 1 # K + 1(mask)
        if prediction:
            input_image = (None, None, 3)
        else:
            input_image = (480,480,3)
        input_heat_mask = KL.Input(shape=(120,120,19), name="mask_heat_input")
        backbone = Backbone(input_image, bck_arch, bck_weights).model
        # if bck_weights == 'imagenet':
        #     backbone.load_weights('/home/igor/PycharmProjects/MultiPoseIdentification/Models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        input_graph = backbone.input
        C2,C3,C4,C5 = backbone.output
        self.fpn_part(C2,C3,C4,C5)
        self.apply_mask(self.D, input_heat_mask)
        self.model = Model(inputs=[input_graph, input_heat_mask], outputs=[self.w])
        print(self.model.summary())

    def fpn_part(self, C2,C3,C4,C5):

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

    def apply_mask(self, x, mask):
        w_name = "weight_masked"

        self.w = KL.Multiply(name=w_name)([x, mask])  # vec_heat



# KeypointNet(18)