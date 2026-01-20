import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ======================
# CONFIG
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
DATASET_PATH = "dataset"

# ======================
# DATA AUGMENTATION (JETSON + SINGLE FRAME SAFE)
# ======================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=180,
    zoom_range=0.3,
    shear_range=0.15,

    brightness_range=(0.75, 1.25),
    channel_shift_range=15.0,   # 🔥 helps RED brick detection

    horizontal_flip=True,
    vertical_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ======================
# BASE MODEL
# ======================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 🔥 Fine-tuning for texture + color
base_model.trainable = True
for layer in base_model.layers[:-25]:
    layer.trainable = False

# ======================
# CLASSIFIER HEAD (STRONG)
# ======================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ======================
# COMPILE
# ======================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# CALLBACKS (CRITICAL)
# ======================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        "rock_classifier_best.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ======================
# TRAIN
# ======================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ======================
# SAVE FINAL MODEL
# ======================
model.save("rock_classifier_jetson.h5")
print("✅ High-accuracy Jetson-ready model saved")