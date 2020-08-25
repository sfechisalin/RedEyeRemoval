import cv2
import numpy as np

class EyeFixer:
    def __init__(self, face_detector, eye_detector):
        self.__face_detector = face_detector
        self.__eye_detector = eye_detector

    def __fill_holes(self, mask):
        maskFloodfill = mask.copy()
        h, w = maskFloodfill.shape[:2]
        maskTemp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
        mask2 = cv2.bitwise_not(maskFloodfill)
        return mask2 | mask

    def __remove_red_eye(self, image,  eye_coordinates):
        imgOut = image.copy()

        for face_eyes in eye_coordinates:
            face_coordinates, eye_coordinates = face_eyes
            x1, y1, w1, h1 = face_coordinates
            face_region = image[y1: y1 + h1, x1: x1 + w1]

            for eye in eye_coordinates:
                x, y, w, h = eye
                # Extract eye from the image.
                eye = face_region[y:y + h, x:x + w]
                # Split eye image into 3 channels
                b = eye[:, :, 0]
                g = eye[:, :, 1]
                r = eye[:, :, 2]

                # Add the green and blue channels.
                bg = cv2.add(b, g)

                # Simple red eye detector
                mask = (r > 150) & (r > bg)

                # Convert the mask to uint8 format.
                mask = mask.astype(np.uint8) * 255

                mask = self.__fill_holes(mask)
                mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

                mean = bg / 2
                mask = mask.astype(np.bool)[:, :, np.newaxis]
                mean = mean[:, :, np.newaxis]
                mean = mean.astype(np.uint8)

                # Copy the eye from the original image.
                eyeOut = eye.copy()


                # Copy the mean image to the output image.
                np.copyto(eyeOut, mean, where=mask)

                # Copy the fixed eye to the output image.
                face_region[y:y + h, x:x + w, :] = eyeOut

            imgOut[y1: y1 + h1, x1: x1 + w1, :] = face_region
            cv2.imshow("output", imgOut)
            return imgOut

    def fix(self, image):
        img_copy = image.copy()
        cv2.imshow("fix_the_following", image)
        without, witha, face_detectiions = self.__face_detector.detect(img_copy)
        cv2.imshow('without', without)
        cv2.waitKey(0)
        cv2.imshow('with', witha)
        cv2.waitKey(0)
        eye_coordinates = self.__eye_detector.detect(witha, face_detectiions)
        return self.__remove_red_eye(image, eye_coordinates)