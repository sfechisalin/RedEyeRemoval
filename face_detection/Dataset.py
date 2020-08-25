from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, image_size, path_dir_positive_input, path_dir_negative_input):
        self.__images = []
        self.__labels = []
        self.__training_images = []
        self.__training_labels = []
        self.__testing_images = []
        self.__testing_labels = []
        self.__image_size = image_size
        self.__path_pos = path_dir_positive_input
        self.__path_neg = path_dir_negative_input
        self.__loadData(self.__path_neg, 'no_face')
        self.__loadData(self.__path_pos, 'face')
        self.__images = np.array(self.__images)
        self.__labels = np.array(self.__labels)
        self.__normalize_data()
        self.__generate_data_for_training_and_testing()

    def __loadData(self, path, label = 'face'):
        print("loading images")
        isFace = 1
        if label != 'face':
            isFace = 0
        count = 0
        for dirName, subdirList, fileList in os.walk(path):
            for f in fileList:
                try:
                    if isfile(join(dirName, f)):
                        # print(join(dirName, f))
                        self.__images.append(cv2.resize(cv2.imread(join(dirName, f), cv2.IMREAD_COLOR), self.__image_size))
                        self.__labels.append(isFace)
                        count = count + 1
                except:
                    print("No image file")
        if isFace == 1:
            print("{} face images".format(count))
        else:
            print("{} no face images".format(count))

    def __normalize_data(self):
        self.__images = np.array(self.__images)
        self.__labels = np.array(self.__labels)
        self.__images = self.__images.astype('float32')
        self.__images /= 255.0

    def __generate_data_for_training_and_testing(self):
        self.__training_images, self.__testing_images, self.__training_labels, self.__testing_labels = train_test_split(self.__images, self.__labels, test_size=.25, random_state=42)
        self.__training_labels = to_categorical(self.__training_labels, num_classes=2)
        self.__testing_labels = to_categorical(self.__testing_labels, num_classes=2)

    def get_training_data(self):
        return self.__training_images, self.__training_labels

    def get_testing_data(self):
        return self.__testing_images, self.__testing_labels