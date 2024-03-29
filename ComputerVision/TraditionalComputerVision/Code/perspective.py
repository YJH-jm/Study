import sys
import numpy as np
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/namecard.jpg")

    if src is None:
        print("Image load failed!")
        sys.exit()

    w, h = 720, 400 # 출력 영상의 크기
    srcQuad = np.array([[325, 307], [760, 369], [718, 611], [231, 515]], np.float32) # 좌측 상단, 우측 상단, 우측 하단, 좌측 하단
    dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)

    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers,(w, h))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()