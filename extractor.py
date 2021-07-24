import cv2
import numpy as np

#imported an img and make it grayscale
frame = cv2.imread("data/test.jpg", cv2.IMREAD_GRAYSCALE)

#used gaussian threshold
f_bw = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        cv2.THRESH_BINARY, 11, 2)

#getting contours and hierarchy(we don't need it here)
cnts, _ = cv2.findContours(f_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#empty image for contours
frame_contours = np.zeros(frame.shape)
#draw the contours
cv2.drawContours(frame_contours, cnts, -1, (0, 255, 0), 3)

cv2.imshow("img", frame_contours)
cv2.waitKey(0)
