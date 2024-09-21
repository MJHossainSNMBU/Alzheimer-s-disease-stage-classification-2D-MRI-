from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

# Define the Custom CNN model
model = Sequential()

# First Convolutional Block
model.add(Conv2D(32, (5, 5), padding='same', input_shape=(IMAGEX, IMAGEY, 3)))  # filter size 5x5 and 32 filters
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))  # max-pooling to reduce feature map size

# Second Convolutional Block
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Third Convolutional Block
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Fourth Convolutional Block
model.add(Conv2D(256, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Flattening and Fully Connected Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))  # First fully connected layer
model.add(Dense(64, activation='relu'))   # Second fully connected layer
model.add(Dropout(0.3))  # Dropout to prevent overfitting

# Output layer
model.add(Dense(3))  # Output layer with 3 nodes (for 3 classes)
model.add(Activation('softmax'))  # Softmax activation for multi-class classification

# Display model summary
model.summary()
