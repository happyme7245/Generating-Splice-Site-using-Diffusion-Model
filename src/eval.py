import one_hot
import deep_cnn
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import f1_score


'''
Model accuracy and loss

Loss: 0.005904489196836948
Accuracy: 0.997972309589386
'''


# define parameters
count = 4
epochs = 50
seq_length = 402
batch_size = 32

# check the relative path of both pos and neg dataset
X, Y = one_hot.readInputs('REAL_dataset/clean_acceptors.pos', 'REAL_dataset/clean_acceptors.neg')

X_train, y_train = X.reshape(-1, seq_length, 4), to_categorical(Y, num_classes=2)

X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)

model = deep_cnn.DeepSplicer(seq_length)

# #train model
# model.fit(X_train_tf, Y_train_tf, epochs=epochs, batch_size=batch_size, verbose=1)

# # save model
# model.save_weights(f'saved_models/deep_splice_model_{count}.h5')

# load model
file_path = f'saved_models/deep_splice_model_{count}.h5'
model = deep_cnn.DeepSplicer(seq_length)
model.load_weights(file_path)

# test
# check the relative path of both pos and neg dataset
X_test, Y_test = one_hot.readInputs(f'SYNTHETIC_dataset/synthetic_{count}.txt', f'SYNTHETIC_dataset/synthetic_test_{count}.txt')
X_test, y_test = X_test.reshape(-1, seq_length, 4), to_categorical(Y_test, num_classes=2)

X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

# predict using pretrained model
loss, accuracy = model.evaluate(X_test_tf, Y_test_tf, verbose=1)
print(f'Loss: {loss} | Accuracy: {accuracy}')

'''
What I obtained

Loss: 434.3860168457031
Accuracy: 0.11170784384012222
???

test set
F1-score: 0.024520667388559243
'''

# Predict using the model
predictions = model.predict(X_test_tf)
predictions_classes = np.argmax(predictions, axis=1)
y_true = np.argmax(Y_test_tf, axis=1)

# Calculate F1-score
f1 = f1_score(y_true, predictions_classes, average='weighted')  # 'weighted', 'macro', 'micro' -> options
print(f'F1-score: {f1}')
