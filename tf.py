import matplotlib
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from dao.UserDAO import UserDAO
from eye_correction.EyeFixer import EyeFixer
from eye_detection.EyeDetector import EyeDetector
from face_detection.Dataset import Dataset
from keras.backend.tensorflow_backend import set_session
from face_detection.utils import convertTuple, pyramid

from face_detection.Model import Model
from face_detection.FaceDetector import FaceDetector
import argparse
import face_detection.utils
import numpy as np
import matplotlib.pyplot as plt


import cv2
import keras
import matplotlib.image as mpimg
import tensorflow as tf

from gui.RedEyeGUI import RedEyeGUI


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model_weights.h5')

def get_statistics(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training ang Validation Accuracy')
    plt.legend()
    # plt.figure()
    plt.savefig("plot1.png")

    plt.clf()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig("plot2.png")
    # plt.show()

def train(modelBuilder, batch_size, epochs,  loss_function, optimizer, metrics, X, Y, x, y):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    np.set_printoptions(threshold=np.inf)

    train_datagen = ImageDataGenerator(rotation_range=30,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,)

    # val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(X, Y, batch_size=batch_size)
    # val_generator = val_datagen.flow(x, y, batch_size=batch_size)

    ntrain = len(X)
    nval = len(x)
    model = modelBuilder.build(loss_function, optimizer, metrics)

    es = EarlyStopping(monitor='val_loss',patience=20,verbose=1, mode='auto')
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=ntrain // batch_size,
                                  epochs=epochs,
                                  validation_data=(x, y),
                                  validation_steps=nval,
                                  callbacks=[es, mc])

    save_model(model)
    get_statistics(history)

def test_model(modelBuilder, loss_function, optimizer, metrics, x, y):
    loaded_model = modelBuilder.load_model_from_file()
    loaded_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    scores = loaded_model.evaluate(x, y, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))


def main(image):
    RedEyeGUI()
    # image_size = (28, 28)
    # input_shape = (28, 28, 3)
    # db = Dataset(image_size, './dataset/face/', './dataset/noface/')
    # modelBuilder = Model(input_shape, construct_model=True)
    # lr = 1e-3
    # epochs = 250
    # batch_size=32
    # loss_function = 'binary_crossentropy' #binary_crossentropy
    # #optimizer = RMSprop(lr=1e-4)#Adam(lr=0.0005)#'adam'
    # optimizer = Adam(lr=lr, decay=lr / epochs)
    # metrics = ['accuracy']
    #
    # # X, Y = db.get_training_data()
    # # x, y = db.get_testing_data()
    # # X = X.reshape([-1, 28, 28, 3])
    # # x = x.reshape([-1, 28, 28, 3])
    # # print(x.shape, y.shape)
    # # print([yy for yy in Y])
    # # print([yy for yy in y])
    # #
    # # train(modelBuilder, batch_size, epochs, loss_function, optimizer, metrics, X, Y, x, y)
    #
    # x = []
    # y = []
    # x.append(cv2.resize(cv2.imread('./test_images/redEye.jpg', cv2.IMREAD_COLOR), image_size))
    # x = np.array(x)
    # x = x.astype('float32')
    # x /= 255.0
    # y.append(1)
    # y = to_categorical(y, num_classes=2)
    # test_model(modelBuilder, loss_function, optimizer, metrics, x, y)
    #
    # loaded_model = modelBuilder.load_model_from_file()
    # #loaded_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    #
    # face_detector = FaceDetector(loaded_model, 1.5, (128, 128), 32, 0.4)
    #
    # eye_detector = EyeDetector()
    #
    # eye_fixer = EyeFixer(face_detector, eye_detector)
    # clean_image = eye_fixer.fix(image)
    #
    # cv2.imshow('clean_image', clean_image)
    # cv2.waitKey(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
    print("My image shape = " + convertTuple(image.shape))

    main(image)