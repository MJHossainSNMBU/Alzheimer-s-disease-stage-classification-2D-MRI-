# Importing necessary libraries and files
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from alzheimernet import AlzheimerNet  # Assuming your model class is defined here
from data_preparation import train_generator, validation_generator  # Assuming data generators are defined here

# Define callbacks for early stopping, model checkpointing, and learning rate reduction
checkpoint = ModelCheckpoint(
    filepath="/content/modelPretrain/jaber_cnn_baseNet.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    verbose=1,
    min_delta=0.0001
)

# Group callbacks into a list
callbacks = [earlystop, checkpoint, reduce_lr]

# Compile the model
model = AlzheimerNet()  # Instantiate the model (assuming it's defined in alzheimernet.py)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(decay=0.01),
    metrics=['accuracy']
)

# Train the model with the defined callbacks and generators
baseNet = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=40,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=callbacks
)

# Save the final model after training
model.save('/content/readymodel/baseNet.h5')
