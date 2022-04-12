import numpy as np
import cv2

img = np.zeros((480, 640), np.uint8)


def on_level_changed(pos):
    global img

    level = pos * 16
    # if level >= 255:
    #     level = 255
    level = np.clip(level, 0, 255)
    img[:, :] = level

    cv2.imshow("image", img)

cv2.namedWindow("image")
cv2.createTrackbar("level", "image", 0, 16, on_level_changed) # image 창 만들어진 후에 

cv2.imshow("image", img)
cv2.waitKey()

cv2.destroyAllWindows()