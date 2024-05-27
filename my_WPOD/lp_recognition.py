import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils
import cv2
from src.data_utils import order_points, convert2Square, draw_labels_and_boxes
from src.char_classification.model import CNN_Model
from skimage.filters import threshold_local
import os
import easyocr
ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

CHAR_CLASSIFICATION_WEIGHTS = 'E:\WorkSpace\Python\Do_An_Tot_Nghiep\yolov9\my_WPOD\weights\lp-ocr\weight.h5'

class LicensePlateRecognizer(object):
    def __init__(self):
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
        self.cnt = 0
        # self.recogChar =easyocr.Reader(['en'])


    def predict(self, image):
        # segmentation
        candidates = self.segmentation(image)

        # recognize characters
        candidates = self.recognizeChar(candidates)

        # format and display license plate
        license_plate_number = self.format(candidates)

        return license_plate_number

    def segmentation(self, LpRegion):
        # apply thresh to extracted licences plate
        # LpRegion = np.array(255*(LpRegion/255)**1.8,dtype ="uint8")
        # V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        gray = cv2.cvtColor(LpRegion, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("V",V)
        # cv2.waitKey(0)
        # adaptive threshold
        # T = threshold_local(V, 15, offset=10, method="gaussian")
        # thresh = (V > T).astype("uint8") * 255
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
        # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
        # cv2.imshow("threash",thresh)
        # cv2.waitKey(0)
        # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)


        candidates = []

        # loop over the unique components
        for label in np.unique(labels):
            # if this is background label, ignore it
            if label == 0:
                continue

            # init mask to store the location of the character candidates
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # find contours from mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                # rule to determine characters
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    # extract characters
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    candidates.append((square_candidate, (y, x)))

        return candidates

    def recognizeChar(self, candidates):
        characters = []
        coordinates = []

        for char, coordinate in candidates:
            characters.append(char)

            coordinates.append(coordinate)


        characters = np.array(characters, dtype=float)

        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:    # if is background or noise, ignore it
                continue
            candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

        return candidates

    def format(self, candidates):
        first_line = []
        second_line = []
        # print(candidates)
        for candidate, coordinate in candidates:

            if candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate
if __name__ == '__main__':
    lp_rec = LicensePlateRecognizer()
    data_img = os.path.join("E:\WorkSpace\Python\Do_An_Tot_Nghiep\Output_wpod_v2\output_16_28_wpod_v2", "img_219.jpg")

    img = cv2.imread(data_img)

    lp_number = lp_rec.predict(img)
    print(lp_number)
