import numpy as np
import cv2 as cv
import time
from multiprocessing import Process, Value, Array

def order_points(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def persp_form(image, pts, m):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    #widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    #widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = int((image.shape[1]-m)/1.75)#max(int(widthA), int(widthB))

    #heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    #heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = int((image.shape[0]-0)/1.6)#max(int(heightA), int(heightB))

    dst1 = [[0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]]
    dst = np.array(dst1, dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def nothing(x):
    pass

def m(qq):
    qq[0] = 5
    

st=time.time()
qq = Array('d', [0, 0, 0])
pr = Process(target=m, args=(qq,), daemon=True)
pr.start()
print(qq)
pr.terminate()
print(time.time()-st)

#cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
#cv.createTrackbar("use", "Tracking", 0, 300, nothing)

while 0:
    m = 135#cv.getTrackbarPos("use", "Tracking")#150
    #a = [(288, 205), (287, 520), (433, 440), (433, 141)]
    img = cv.imread("screenshoot-t0.png")
    #st = time.time()
    img = cv.copyMakeBorder(img, 0,0,m,m, cv.BORDER_CONSTANT, value=(255,255,255))
    a = [(m, 0), (0, img.shape[0]), (img.shape[1], img.shape[0]), (img.shape[1]-m, 0)]
    some = persp_form(img.copy(),np.array(a),m)
    #print(time.time()-st)

    for i in range(0-5*5,some.shape[1]+1,5):
        cv.line(some, (i,0), (i,some.shape[0]), (0,0,255), 1)
    for i in range(0,some.shape[0]+1,5*8):
        cv.line(some, (0,i), (some.shape[1],i), (0,0,255), 1)

    cv.namedWindow("some", cv.WINDOW_NORMAL)
    cv.imshow("img1",img)
    cv.imshow("some",some)
    cv.imwrite("resul.png",some)
    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break