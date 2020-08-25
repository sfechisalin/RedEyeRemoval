from face_detection.utils import rgb2gray, pyramid, sliding_window, non_max_suppression
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array

class FaceDetector:
    def __init__(self, model, downscale = 1.5, window_size = (128, 128), window_step_size = 32, threshold=0.4):
        self.__model = model
        self.__downscale = downscale
        self.__window_size = window_size
        self.__window_step_size = window_step_size
        self.__threshold = threshold

    def detect(self, image):
        # image /= 255

        cloned_image = image.copy()

        # image = rgb2gray(image)

        detections = []

        depth_level = 0

        max_area = 200000
        coordinates  = (0, 0, 0, 0)
        for scaled_image in pyramid(image, scale=1.5):
            # cv2.imshow('image_pyramid', scaled_image)
            # cv2.waitKey(0)
            # if scaled_image.shape[0] < self.__window_size[0] or scaled_image.shape[1] < self.__window_size[1]:
            #     break
            for (x, y, window) in sliding_window(scaled_image, self.__window_step_size, self.__window_size):
                if window.shape[0] != self.__window_size[0] or window.shape[1] != self.__window_size[0]:
                    continue
                # if depth_level >= 2:
                #     print("HEREEEEEEEEEEEEEEEE")
                #     cv2.imshow('image', window)
                #     cv2.waitKey(0)
                window = cv2.resize(window, (28, 28))
                #window = window.astype("float") / 255.0
                # window = window.reshape([-1, 28, 28, 3])
                # test = np.array(window)
                # test = test.astype('float32')
                # test /= 255.0
                # test = np.expand_dims(test, axis=0)
                # test_image = test.reshape([-1, 28, 28, 3])
                #
                # prediction = self.__model.predict(test_image)

                window = window.astype("float") / 255.0
                window = img_to_array(window)
                window = np.expand_dims(window, axis=0)
                prediction = self.__model.predict(window)
                print(prediction)

                if prediction[0][1] > 0.9:
                    # cv2.rectangle(image, (x, y), (x + 128, y + 128), (255, 255, 0), 2)
                    x1 = int(x * (self.__downscale ** depth_level))
                    y1 = int(y * (self.__downscale ** depth_level))
                    # x2 = x1 + int(self.__window_size[0] * (
                    #                            self.__downscale ** depth_level))
                    # y2 = y1 + int(self.__window_size[1] * (
                    #         self.__downscale ** depth_level))
                    x2 = int((x + self.__window_size[0]) * (self.__downscale ** depth_level))
                    y2 = int((y + self.__window_size[1]) * (self.__downscale ** depth_level))
                    detections.append((x1, y1,
                                       x2, y2))

                    if (x2 - x1) * (y2 - y1) < max_area:
                        max_area = (x2 - x1) * (y2 - y1)
                        coordinates = (x1, y1, x2, y2)
            depth_level += 1

        clone_before_nms = cloned_image.copy()
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(clone_before_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        detections = non_max_suppression(np.array(detections), self.__threshold)

        clone_after_nms = cloned_image
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(clone_after_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        x1, y1, x2, y2 = coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.imshow('my image', image)
        cv2.waitKey(0)

        return clone_before_nms, clone_after_nms, detections