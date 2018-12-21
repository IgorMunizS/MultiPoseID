from keras import backend
from keras import layers
from keras.layers import Input
from keras.models import Model
from .resnet import ResNet50, ResNet101, ResNeXt50, ResNeXt101

class Backbone():
    # initialize model from config file
    def __init__(self, input=None, architecture='resnet50', weights=None):

        # hyperparameter config file
        if input == None:
            self.input = (None, None, 3)
        else:
            self.input = input
        # create Keras model
        self.model = self._create_model(architecture, weights)

    # creates keras model
    def _create_model(self, architecture, weights):

        assert architecture in ["resnet50", "resnet101", "resnetx50", "resnetx101"]

        input_image = Input(shape=self.input,
                            name="input_1")


        if architecture == "resnet50":
            model = ResNet50(False, weights, input_image, None, None)

        if architecture == "resnet101":
            model = ResNet101(False, weights, input_image, None, None)

        if architecture == "resnetx50":
            model = ResNeXt50(False, weights, input_image, None, None)

        if architecture == "resnetx101":
            model = ResNeXt101(False, weights, input_image, None, None)

        return model



# a = Backbone(architecture="resnet50").model
# a.load_weights("../Models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")