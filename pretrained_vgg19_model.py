
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model

# Assuming IMAGE_SIZE and number of folders (classes) are defined elsewhere
# Example for IMAGE_SIZE: IMAGE_SIZE = [224, 224] if not yet defined
# Example for len(folders): len(folders) = 3 (for 3-class classification)

# Load the VGG19 model pre-trained on ImageNet, excluding the top layer
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
vgg.summary()

# Freeze all layers except the last 3 layers for training
for layer in vgg.layers[:-3]:
    layer.trainable = False

# Summary to show which layers are trainable
vgg.summary()

# Adding custom layers on top of VGG19
x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.25)(x)
prediction = Dense(len(folders), activation='softmax')(x)  # Adjust based on number of classes

# Create the final model
model_n = Model(inputs=vgg.input, outputs=prediction)
model_n.summary()

# Visualizing trainable layers
for layer in model_n.layers:
    print(layer, layer.trainable)
