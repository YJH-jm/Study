import sys
import numpy as np
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/cat.bmp")

    if src is None:
        print("Image load failed!")
        sys.exit()

    rc = (250, 120, 200, 200) # retangle tuple (w, y, w, h)

    # 원본 영상에 그리기
    cpy = src.copy()
    cv2.rectangle(cpy, rc, (0, 0, 255), 2) 
    cv2.imshow("src", cpy)
    cv2.waitKey()

    # 피라미드 영상에 그리기

    for i in range(1, 4):
        src = cv2.pyrDown(src)
        cpy = src.copy()
        cv2.rectangle(cpy, rc, (0, 0, 255), 2, shift=i) # 이미지 사이즈가 작아져도 rc가 변하지 않음
        # shift : 그림을 그리기 위한 좌표를 가로,세로 만큼 얼마나 줄일 건지를 나타냄 , 사용을 권장하지는 않음
        cv2.imshow("src", cpy)
        cv2.waitKey()

    cv2.destroyAllWindows()
