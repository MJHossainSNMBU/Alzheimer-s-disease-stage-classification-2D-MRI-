from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training, validation, and test set generation

# Define ImageDataGenerator for training, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizing the images
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define generators for training, validation, and test datasets
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMAGEX, IMAGEY),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    valid_path,
    target_size=(IMAGEX, IMAGEY),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMAGEX, IMAGEY),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)
