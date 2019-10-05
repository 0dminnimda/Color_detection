import numpy as np
import cv2 as cv
from mss import mss
import time

def nothing(x):
    pass

def gtbp(name):
    out_name = []
    for i in name:
        out_name.append(cv.getTrackbarPos(i, "Tracking"))
    return out_name

def stbp(name):
    for i,j in name.items():
        cv.setTrackbarPos(i,"Tracking",int(j))

def ctb(name):
    for i in name:
        cv.createTrackbar(i[0], "Tracking", i[1], i[2], nothing)

cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
cr_tb = [["LH", 110, 255],["LS", 100, 255],["LV", 100, 255],["UH", 164, 255],["US", 255, 255],["UV", 255, 255]]
cr_tb2 = [["LH2", 0, 255],["LS2", 0, 255],["LV2", 0, 255],["UH2", 0, 255],["US2", 0, 255],["UV2", 255, 255]]
cr_tb3 = [["LH3", 0, 255],["LS3", 255, 255],["LV3", 111, 255],["UH3", 0, 255],["US3", 255, 255],["UV3", 255, 255]]
ctb(cr_tb3)
ctb(cr_tb2)
ctb(cr_tb)

left = 5 # 9
top = 45 # 40
wid = 1290 - left # 1294
hei = 770 - top

bbox = {'top': top, 'left': left, 'width': wid, 'height': hei}

sct = mss()

cap = cv.VideoCapture('2019-10-06 02-16-00.mp4')
cap2 = cv.VideoCapture('2019-10-06 02-17-30.mp4')

#b_cascade = cv.CascadeClassifier('cascade2.xml')
b1_cascade = cv.CascadeClassifier('cascadelbp.xml')
#start = time.time()
nn = 0

while 1:
    #_, frame = cap.read()

    #print(type(ret),type(frame))

    sct_img = sct.grab(bbox)
    img1 = np.array(sct_img)
    frame = img1
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #print(np.array(sct_img))

    name = ["LH","LS","LV","UH","US","UV"]
    name2 = ["LH2","LS2","LV2","UH2","US2","UV2"]
    name3 = ["LH3","LS3","LV3","UH3","US3","UV3"]
    l_h, l_s, l_v, u_h, u_s, u_v = gtbp(name)
    l_h2, l_s2, l_v2, u_h2, u_s2, u_v2 = gtbp(name2)
    l_h3, l_s3, l_v3, u_h3, u_s3, u_v3 = gtbp(name3)

    l_b, u_b = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
    l_b2, u_b2 = np.array([l_h2, l_s2, l_v2]), np.array([u_h2, u_s2, u_v2])
    l_b3, u_b3 = np.array([l_h3, l_s3, l_v3]), np.array([u_h3, u_s3, u_v3])

    mask = cv.inRange(hsv, l_b, u_b)
    #res = cv.bitwise_and(frame, frame, mask=mask)
    mask2 = cv.inRange(hsv, l_b2, u_b2)
    #res2 = cv.bitwise_and(frame, frame, mask=mask2)
    mask3 = cv.inRange(hsv, l_b3, u_b3)
    res3 = cv.bitwise_and(frame, frame, mask=mask3)
    bet_mask = cv.bitwise_or(mask2, mask)
    bet_mask2 = cv.bitwise_or(bet_mask, mask3)
    bet = cv.bitwise_or(frame, frame, mask=bet_mask2)

    gray = cv.cvtColor(bet, cv.COLOR_BGR2GRAY)
    #print(sct_img)

    #boxes = b_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in boxes:
    #    cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    #    roi_gray = gray[y:y+h, x:x+w]
    #    roi_color = img1[y:y+h, x:x+w]

    boxes1 = b1_cascade.detectMultiScale(gray, 1.3, 8)
    for (x,y,w,h) in boxes1:
        cv.rectangle(res3,(x,y),(x+w,y+h),(0,255,0),2)
        #roi_gray1 = gray[y:y+h, x:x+w]
        roi_color1 = bet[y:y+h, x:x+w]

    #cv.namedWindow ( "res" , cv.WINDOW_NORMAL)
    #cv.imshow("frame", frame)
    #cv.imshow("res", res)
    #cv.imshow("res2", res2)
    #cv.imshow("res3", res3)
    cv.imshow("bet", bet)
    #cv.imshow("img", img1)

    if cv.waitKey(3) & 0xFF == ord('4'):
        cv.imwrite(f"screenshoot{nn+1}.png", bet)
        cv.imwrite(f"screenshoot{nn+1}.jpg", bet)
        nn += 1

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break