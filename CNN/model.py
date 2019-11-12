import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
    batch_size = 128
    num_classes = 10
    epochs = 12

    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    img_rows, img_cols = 28, 28
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)
    input_shape = (img_rows, img_cols, 1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(strides=1))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adagrad(),
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    model_json = model.to_json()
    print(score)
    with open("/Users/ducnguyen/python/untitled2/CNN/models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("/Users/ducnguyen/python/untitled2/CNN/models/weights.h5")

def build_model():
    json_file = open("/Users/ducnguyen/python/untitled2/CNN/models/model.json", 'r')
    json_config = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights("/Users/ducnguyen/python/untitled2/CNN/models/weights.h5")
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adagrad(),
                  metrics=['accuracy'])
    return model

class Model:
    def __init__(self):
        self.model = build_model()

    def classify(self, imgarray):
        imgarray = imgarray.reshape((28, 28, 1))
        imgarray = np.expand_dims(imgarray, axis=0)
        imgarray/=255
        predictions = self.model.predict(imgarray)
        return np.argmax(predictions)

if __name__ == "__main__":
    main()