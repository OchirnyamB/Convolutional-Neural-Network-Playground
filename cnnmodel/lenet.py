from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be channes last
        # Depth is the number of channels in the input image
        model = Sequential()
        inputShape = (height, width, depth)

        # If we are using channels first, update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # First set of CONV => RELU => POOL Layers
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Second set of CONV => RELU => POOL Layers
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        # First and only set of FC => RELU Layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Final softmax classifier with class probabilities
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model

        
