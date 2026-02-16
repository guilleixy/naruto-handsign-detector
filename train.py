import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

TRAIN_DATA = "./data"
TEST_DATA = "./data"
BATCH_SIZE = 16
TRAIN_SIZE = 4777
TEST_SIZE = 300

def get_datagen(dataset, augmented=False):
    if augmented:
        datagen = ImageDataGenerator(
            rescale=1./255,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=25,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=False,
            brightness_range=[0.8, 1.1]
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        
    return datagen.flow_from_directory(
        dataset,
        target_size=(224, 244),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )

train_generator = get_datagen(TRAIN_DATA, augmented=True)
test_generator = get_datagen(TEST_DATA, augmented=False)

def load_model(model):
    # MobileNet Model
    if model == "MN":
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(224, 244, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet',
            pooling='None'
        
        )
        # Freeze every layer except the last one   
        for layer in model.layers[:-1]:
            layer.trainable = False

        mobile_net = Flatten()(model.output)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(4096, activation='relu')(mobile_net)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(1024, activation='relu')(mobile_net)
        mobile_net = Dropout(0.3)(mobile_net)
        mobile_net = Dense(12, activation='softmax')(mobile_net)

        mobile_net_mobile = Model(model.input, mobile_net, name='Altered_MobileNet')
        mobile_net_mobile.summary()

        model = mobile_net_mobile

    return model
model = load_model('MN')

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.5, patience=10, min_lr=0.001, verbose=1)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                        mode='auto', baseline=None, restore_best_weights=True)
model.compile(loss='categorical_crossentropy',
                optimizer=adam, metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=test_generator, 
    steps_per_epoch=TRAIN_SIZE// BATCH_SIZE,
    validation_steps=TEST_SIZE// BATCH_SIZE,
    shuffle=True,
    epochs=50,
    callbacks=[early_stopper],
    #use_multiprocessing=False,
)

model.save("./MN_Naruto_Model.keras")