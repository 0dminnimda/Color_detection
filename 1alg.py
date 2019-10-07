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

name = ["LH","LS","LV","UH","US","UV"]
name2 = ["LH2","LS2","LV2","UH2","US2","UV2"]
name3 = ["LH3","LS3","LV3","UH3","US3","UV3"]
l_h, l_s, l_v, u_h, u_s, u_v = gtbp(name)
l_h2, l_s2, l_v2, u_h2, u_s2, u_v2 = gtbp(name2)
l_h3, l_s3, l_v3, u_h3, u_s3, u_v3 = gtbp(name3)

l_b, u_b = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
l_b2, u_b2 = np.array([l_h2, l_s2, l_v2]), np.array([u_h2, u_s2, u_v2])
l_b3, u_b3 = np.array([l_h3, l_s3, l_v3]), np.array([u_h3, u_s3, u_v3])

left = 5 # 9
top = 45 # 40
wid = 1290 - left # 1294
hei = 770 - top

bbox = {'top': top, 'left': left, 'width': wid, 'height': hei}

sct = mss()

cap = cv.VideoCapture('2019-10-06 02-16-00.mp4')
cap2 = cv.VideoCapture('2019-10-06 02-17-30.mp4')
cap_new = cv.VideoCapture('2019-10-07 12-58-00.mp4')#2019-10-07 12-45-36.mp4')#2019-10-07 12-06-37.mp4')
cap_new.set(cv.CAP_PROP_FPS, 5)
#hei = cap_new.get(cv.CAP_PROP_FRAME_HEIGHT)
#wid = cap_new.get(cv.CAP_PROP_FRAME_WIDTH)
#cap_new.set(cv.CAP_PROP_FRAME_WIDTH , 1024)
#cap_new.set(cv.CAP_PROP_FRAME_HEIGHT , 720)  

#b_cascade = cv.CascadeClassifier('cascade2.xml')
#b1_cascade = cv.CascadeClassifier('cascadelbp.xml')
#start = time.time()
nn = 0
m=0

imggg = cv.imread("screenshoot10.png")

while 1:

    #ret, frame = cap_new.read()
    #print(type(ret),type(frame))

    #sct_img = sct.grab(bbox)
    #img1 = np.array(sct_img)
    frame = imggg#img1
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    

    mask = cv.inRange(hsv, l_b, u_b)
    #res = cv.bitwise_and(frame, frame, mask=mask)
    mask2 = cv.inRange(hsv, l_b2, u_b2)
    #res2 = cv.bitwise_and(frame, frame, mask=mask2)
    mask3 = cv.inRange(hsv, l_b3, u_b3)
    res3 = cv.bitwise_and(frame, frame, mask=mask3)
    bet_mask = cv.bitwise_or(mask2, mask)
    bet_mask2 = cv.bitwise_or(bet_mask, mask3)
    bet = cv.bitwise_or(frame, frame, mask=bet_mask2)

    #gray = cv.cvtColor(bet, cv.COLOR_BGR2GRAY)
    #print(sct_img)

    #boxes = b_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in boxes:
    #    cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    #    roi_gray = gray[y:y+h, x:x+w]
    #    roi_color = img1[y:y+h, x:x+w]

    #boxes1 = b1_cascade.detectMultiScale(gray, 1.3, 8)
    #for (x,y,w,h) in boxes1:
    #    cv.rectangle(res3,(x,y),(x+w,y+h),(0,255,0),2)
    #    #roi_gray1 = gray[y:y+h, x:x+w]
    #    roi_color1 = bet[y:y+h, x:x+w]

    imgray = cv.cvtColor(res3, cv.COLOR_BGR2GRAY)
    rett, thresh = cv.threshold(imgray, 127, 0, 0)
    im2, contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    re3 = cv.drawContours(res3.copy(),
                         contours,
                        -1,
                       (0,255,0),
                       3)
    cv.imshow("bet", re3)
    print(im2,contours)
    #a, fc = cv.findContours( mask3.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    a1, fc2 = cv.findContours( mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # внешние контуры
    #a2, fc3 = cv.findContours( mask3.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    #a3, fc4 = cv.findContours( mask3.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # многоуровневая иерархия
    #a3, fc4 = cv.findContours( mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #dc = cv.drawContours( bet.copy(), a, -1, (0, 150, 0))
    dc2 = cv.drawContours( bet.copy(), a1, 0, (0, 150, 0), 3)
    #dc3 = cv.drawContours( bet.copy(), a2, -1, (0, 150, 0))
    #dc4 = cv.drawContours( bet.copy(), a3, -1, (0, 150, 0))
    print(a1, fc2)
    #if a1 != None:
        #for i in fc2:
    bet_cont = cv.bitwise_or(bet, bet, mask=mask3)
    moments = cv.moments(mask3.copy(), 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    # будем реагировать только на те моменты,
    # которые содержать больше 100 пикселей
    if dArea > 75:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
        cv.circle(dc2, (x, y), 10, (255,0,255), -1)

    #cv.imshow('result', dc2)

    #cv.namedWindow ( "res" , cv.WINDOW_NORMAL)
    #cv.imshow("frame", frame)
    #cv.imshow("res", res)
    #cv.imshow("res2", res2)
    #cv.imshow("res3", res3)
    #cv.imshow("bet", bet)
    #cv.imshow("bet1", dc)
    cv.imshow("bet2", dc2)
    #cv.imshow("bet3", dc3)
    #cv.imshow("bet4", dc4)
    #cv.imshow("img", img1)

    if cv.waitKey(3) & 0xFF == ord('4'):
        cv.imwrite(f"screenshoot{nn+10}.png", bet)
        #cv.imwrite(f"screenshoot{nn+10}.jpg", bet)
        nn += 1

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break

cap.release()