import imutils
import cv2
import numpy as np
from imutils import contours
from os.path import join
from time import time
import json
from tensorflow.keras.models import load_model


def _show_image(win, img, destroy=True):
    cv2.imshow(win, img)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyWindow(win)

def read_digits(gray, image, ocr_model, skew=9, avg_digit_width=16, brightness=70, debug=False):
    _, gray = cv2.threshold(gray, brightness, 255, 0)  # extract white digits
    gray_inv = cv2.bitwise_not(gray)   # turn to black digits
    if debug:
        cv2.imwrite('save/gray_inv.jpg', gray_inv)

    # to locate digits area
    cnts = cv2.findContours(gray_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    largest_area = sorted(cnts, key=cv2.contourArea)[-1]  # find LCD = the largest white background
    mask = np.zeros(image.shape, np.uint8)  # all black in mask
    cv2.drawContours(mask, [largest_area], 0, (255, 255, 255), -1)  # make roi area white in mask
    output = cv2.bitwise_and(image, mask)  # pick roi from image
    if debug:
        cv2.imwrite('save/output.jpg', output)
    roi_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)  # white digits
    _, roi_gray = cv2.threshold(roi_gray, brightness, 255, 0)  # highlight white digits
    if debug:
        cv2.imwrite('save/roi_gray.jpg', roi_gray)


    # to find each digit
    thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if debug:
        cv2.imwrite('save/thresh.jpg', thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    (x, y, w, h) = cv2.boundingRect(cnts[0])
    roi_small = thresh[y:y + h, x:x + w]
    warped = roi_small.copy()  # black digits
    if debug:
        cv2.imwrite('save/roi_small.jpg', roi_small)


    # the numbers displayed a little bit leaning to right side, to make them upright
    height, width = warped.shape
    width -= 1
    height -= 1
    rect = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]], dtype="float32")
    dst = np.array([
        [-skew, 0],
        [width - skew, 0],
        [width, height],
        [0, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(warped, M, (width + 1, height + 1))
    if debug:
        cv2.imwrite('save/warped.jpg', warped)
    output = cv2.warpPerspective(output, M, (width + 1, height + 1))

    # segment 2 and segment 5 separated so we do vertical dilate and erode to connect them
    vert_dilate3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 3))
    dilation = cv2.dilate(warped, vert_dilate3)
    dilation = cv2.erode(warped, vert_dilate3)  # black digits
    dilation_inv = cv2.bitwise_not(dilation)  # white digits
    if debug:
        cv2.imwrite('save/dilation_inv.jpg', dilation_inv)

    # locate each digit
    cnts = cv2.findContours(dilation_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    digitCnts = []
    # loop over the digit area candidates
    for _c, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if debug:
            print("Contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))

        if 10 <= h <= 35 and w <= 20 and y < 20:
            digitCnts.append(c)
    # sort the contours from left-to-right, then initialize the actual digits themselves
    # avoid error: ValueError: not enough values to unpack (expected 2, got 0)
    try:
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    except ValueError:
        return [-1]
    if debug:
        print("Found {} ROIs".format(len(digitCnts)))

    digits = []
    # loop over each of the digits
    for _c, c in enumerate(digitCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if debug:
            print("Selected contour {}: w={}, h={}, x={}, y={}".format(_c, w, h, x, y))
        # manually override the width of number 1
        if w < 9:
            x -= avg_digit_width - w
            w = avg_digit_width
            if debug:
                print("  changed contour : w={}, h={}, x={}, y={}".format(w, h, x, y))
        elif w > avg_digit_width + 1:
            w = avg_digit_width
            point = _c
        roi = dilation_inv[y:y + h, x:x + w]
        cv2.imwrite("save/image_{}.jpg".format(_c), roi)


        ht, wd = roi.shape
        ww = 32
        hh = 32
        color = (0, 0, 0)
        result = np.zeros((hh, ww), dtype=np.uint8)
        # result = np.full((hh, ww, 3), color, dtype=np.uint8)
        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2
        # copy img image into center of result image
        result[yy:yy + ht, xx:xx + wd] = roi
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        img = result[np.newaxis, :, :, np.newaxis]
        # X_test = np.pad(image, ((0,0),(2,2),(2,2),(0,0)), 'constant')


        result = ocr_model.predict(img)
        digits.append(result.argmax())
    digits.insert(-3, '.')

    return digits


if __name__ == '__main__':
    """
    CNN
    """
    path = 'save'
    image = join(path, 'f08.jpg')

    image = cv2.imread(image)

    image = cv2.resize(image, (128, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    model_name = 'model/OCR'
    ocr_model = load_model(model_name)
    show_image = True
    digits = read_digits(gray, image, ocr_model, skew=8, avg_digit_width=13, brightness=70, debug=show_image)
    print(digits)