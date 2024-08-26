import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf


def DeepSplicer(length):

    model = tf.keras.models.Sequential(name='DeepSplicer')

    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
