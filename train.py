import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=(100, 100),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=(100, 100),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes found:", class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(100, 100, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("fruit_model.h5")
print("Model saved as fruit_model.h5")