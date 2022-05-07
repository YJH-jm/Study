import cv2
import sys
import numpy as np



def getGrayHistoImage(hist):
    imgHist = np.full((100, 256), 255, dtype=np.uint8)

    histMax = np.max(hist)
        
    for x in range(256):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist

if __name__ == "__main__":
            

    src = cv2.imread("images/Hawkes.jpg", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        sys.exit()

    dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)

    
    gmin = np.min(src)
    gmax = np.max(src)
    dst = np.clip((src-gmin) * 255. / (gmax-gmin), 0, 255).astype(np.uint8)


    hist = cv2.calcHist([src], [0], None, [256], [0, 256]) # histogram 계산하는 함수
    histImg = getGrayHistoImage(hist)

    hist2 = cv2.calcHist([dst], [0], None, [256], [0, 256]) # histogram 계산하는 함수
    histImg2 = getGrayHistoImage(hist2)
    
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    
    cv2.imshow("histImg", histImg)
    cv2.imshow("histImg2", histImg2)

    # 2.



    cv2.waitKey()

    cv2.destroyAllWindows()

