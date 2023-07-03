import sys
import numpy as np
import cv2

def drawROI(img, corners):
    cpy = img.copy()
    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[0].astype(int)),  tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)),  tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)),  tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)),  tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0) # 입력 영상에 합성
    
    return disp

def onMouse(event, x, y, flags, param): # (x, y) -> 클릭한 점의 좌표
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow('img', cpy)
                ptOld = (x, y)
                break



src = cv2.imread("./images/scanned.jpg")
if src is None:
    print("Image load failed!")
    sys.exit()

h, w = src.shape[:2]
dw = 500
dh = round(dw * 297/210) # A4 용지의 크긱 : 210x297cm

# 모서리 점들의 좌표, 드레그 상태 여부
srcQuad = np.array([[30, 30], [30, h-30], [w-30, h-30], [w-30,30]], np.float32) # 임의의 초기값
dstQuad = np.array([[0,0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)
dragSrc = [False, False, False, False]

# 모서리점, 사각형 그리기 
disp = drawROI(src, srcQuad)
cv2.imshow("img", disp)
cv2.setMouseCallback("img", onMouse)

while True:
    key = cv2.waitKey()
    if key == 13: # Enter
        break
    elif key == 27: # ESC
        cv2.destroyWindow("img")
        sys.exit()

# 투시 변환
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
dst = cv2.warpPerspective(src, pers, (dw,dh), flags=cv2.INTER_CUBIC)

cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()