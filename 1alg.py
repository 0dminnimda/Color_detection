import numpy as np
import cv2 as cv
from mss import mss
import time
#from PIL import Image

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
cr_tb = [["LH", 0, 255],["LS", 0, 255],["LV", 0, 255],["UH", 255, 255],["US", 255, 255],["UV", 255, 255]]
cr_tb2 = [["LH2", 0, 255],["LS2", 0, 255],["LV2", 0, 255],["UH2", 255, 255],["US2", 255, 255],["UV2", 255, 255]]
ctb(cr_tb2)
ctb(cr_tb)

br_guy = {"LH":110,"LS":100,"LV":100,"UH":164}
block = {"UH2":0,"US2":0}
stbp(br_guy)
stbp(block)

left = 5 # 9
top = 45 # 40
wid = 1290 - left # 1294
hei = 770 - top # 770

bbox = {'top': top, 'left': left, 'width': wid, 'height': hei}

sct = mss()

#b_cascade = cv.CascadeClassifier('cascade2.xml')
#b1_cascade = cv.CascadeClassifier('cascade.xml')
#start = time.time()

while 1:

    #start = time.time()

    sct_img = sct.grab(bbox)
    #print(np.array(sct_img))
    img1 = np.array(sct_img)
    #gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    #print(sct_img)

    #end = time.time()
    #print(str(sct_img) + ' image capture time(ms):', (end - start)*100)

    #boxes = b_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in boxes:
    #    cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    #    roi_gray = gray[y:y+h, x:x+w]
    #    roi_color = img1[y:y+h, x:x+w]

    #boxes1 = b1_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in boxes1:
    #    cv.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
    #    roi_gray1 = gray[y:y+h, x:x+w]
    #    roi_color1 = img1[y:y+h, x:x+w]

    
    frame = img1

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    name = ["LH","LS","LV","UH","US","UV"]
    name2 = ["LH2","LS2","LV2","UH2","US2","UV2"]
    l_h, l_s, l_v, u_h, u_s, u_v = gtbp(name)
    l_h2, l_s2, l_v2, u_h2, u_s2, u_v2 = gtbp(name2)

    # 94 106 104 255 255 255

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    l_b2 = np.array([l_h2, l_s2, l_v2])
    u_b2 = np.array([u_h2, u_s2, u_v2])

    mask = cv.inRange(hsv, l_b, u_b)
    res = cv.bitwise_and(frame, frame, mask=mask)
    mask2 = cv.inRange(hsv, l_b2, u_b2)
    res2 = cv.bitwise_and(frame, frame, mask=mask2)
    bet_mask = cv.bitwise_or(mask2, mask)
    bet = cv.bitwise_or(frame, frame, mask=bet_mask)

    #cv.namedWindow ( "res" , cv.WINDOW_NORMAL)
    #cv.imshow("frame", frame)
    #cv.imshow("mask", mask)
    #cv.imshow("frame", hsv)
    #cv.imshow("res", res)
    #cv.imshow("res2", res2)
    cv.imshow("bet", bet)


    #cv.imshow('screen', img1)

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break