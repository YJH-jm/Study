import sys
import numpy as np
import cv2

oldx = oldy = -1

def on_mouse(event, x, y, flags, param):

    global img, oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        print("EVENT_LBUTTONDOWN: {}, {}".format(x, y))
        oldx, oldy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        print("EVENT_LBUTTONUP : {}, {}".format(x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON: # == 말고 &로 하는 것이 좋음 , == 는 다른 버튼과 같이 눌린 것을 판단 할 수 없음
            # print("EVENT_MOUSEMOVE : {}, {}".format(x, y))
            # cv2.circle(img ,(x, y), 5, (0, 0, 225), -1)  # 빠르게 마우스를 움직이면 선이 끊어지게 출력
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 225), 5, cv2.LINE_AA) 
            cv2.imshow("image", img)
            oldx, oldy = x, y


img = np.ones((480, 640, 3), dtype=np.uint8) * 255

# cv2.namedWindow("image")
cv2.imshow("image", img)
cv2.setMouseCallback("image", on_mouse)
cv2.waitKey()

cv2.destroyAllWindows()