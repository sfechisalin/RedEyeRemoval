from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU


class Model:
    def __init__(self, input_shape, construct_model=True):
        self.__input_shape = input_shape
        self.__model = Sequential()
        if construct_model == True:
            self.__constructModel(input_shape, 2)

    def __constructModel(self, input_shape, classes):
            # self.__model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 1), padding='same'))
            # self.__model.add(MaxPooling2D((2, 2)))
            # self.__model.add(Conv2D(64, (3, 3), activation='relu'))
            # self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            # self.__model.add(Conv2D(128, (3, 3), activation='relu'))
            # self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            # self.__model.add(Conv2D(128, (3, 3), activation='relu'))
            # self.__model.add(MaxPooling2D((2, 2)))
            # self.__model.add(Flatten())
            # self.__model.add(Dropout(0.5)) #Dropout for regularization
            # self.__model.add(Dense(512, activation='relu'))
            # self.__model.add(Dense(1, activation='sigmoid'))
            # first set of CONV => RELU => POOL layers
            self.__model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
            self.__model.add(Conv2D(32, (3, 3), activation='relu'))
            self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model.add(Dropout(0.25))
            # second set of CONV => RELU => POOL layers
            self.__model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            self.__model.add(Conv2D(64, (3, 3), activation='relu'))
            self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model.add(Dropout(0.25))
            # third set of CONV => RELU => POOL layers
            self.__model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            self.__model.add(Conv2D(64, (3, 3), activation='relu'))
            self.__model.add(MaxPooling2D(pool_size=(2, 2)))
            self.__model.add(Dropout(0.25))
            # first (and only) set of FC => RELU layers and softmax classifier
            self.__model.add(Flatten())
            self.__model.add(Dense(512, activation='relu'))
            self.__model.add(Dropout(0.5))
            self.__model.add(Dense(classes, activation='softmax'))

    def save_model(self):
        model_json = self.__model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.__model.save_weights("model1.h5")
        print("Saved model to disk")

    def load_model_from_file(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_weights.h5")
        print("Loaded model from disk")
        return loaded_model

    def build(self, loss_function, optimizer, metrics):
        self.__model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
        return self.__model

