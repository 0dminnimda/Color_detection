import numpy as np
import cv2 as cv
from mss import mss
import time
from pynput import mouse
import os
#from pynput import keyboard
from pynput.mouse import Button, Controller
import math as ma
#from mose1 import map_count, shoot, walk, end

def nothing(x):
    pass
    pass

def map_count(ab1, ac1, n, nn):
    #ab1, ac1 = input().split()
    #ab1, ac1 = int(ab1), int(ac1)
    if nn == True:
        ab1 += 3
        ac1 += 40
        ab1, ac1 = (-1295+int(ab1)), (-770+int(ac1))
    #n = 60
    abb = ab1
    acc = ac1
    while ab1 > n or ac1 > n:
        ab1 /= 2
        ac1 /= 2
    ab1 = ma.fabs(ab1)
    ac1 = ma.fabs(ac1)
    
    if ab1 > ac1:
        ac = (n*ac1)/ab1
        ab = n
    elif ab1 < ac1:
        ab = (n*ab1)/ac1
        ac = n
    elif ab1 == ac1:
        ac, ab = n, n

    if acc < 0:
        ac = -ac
    if abb < 0:
        ab = -ab

    return ab, ac

def shoot(mou, dirX, dirY, rangee, nn = False):
    mou.position = (1200, 650)
    t = 0.1**1.125
    #n = 20
    dirX, dirY = map_count(dirX, dirY, rangee, nn)
    #pos = mou.position
    mou.press(Button.left)
    #for i in range(n):
        #print(mou.position)
    time.sleep(t/1.25)#/n)
    mou.move(dirX, dirY)#/n)
    time.sleep(t/1.25)#/n)
    mou.release(Button.left)
    #mou.position = pos
    #time.sleep(0.234)
    pass

def walk(mou, dirX, dirY, rangee, tim, nn = False):
    mou.position = (150, 660)
    t = 0.1**1.125
    dirX, dirY = map_count(dirX, dirY, rangee, nn)
    mou.press(Button.left)
    time.sleep(t/2)
    mou.move(dirX, dirY)
    time.sleep(tim)
    mou.release(Button.left)

def start(mou, an, bn):
    # разворачивание окна
    mou.position = (460, 1400)
    mou.click(Button.left, 1)

    time.sleep(0.1)
    # начало игры
    mou.position = (an, bn)
    mou.click(Button.left, 1)

def end(mou, an, bn):
    time.sleep(2.5)
    # скиншот
    #mou.position = (1225, 795)
    #mou.click(Button.left, 1)

    # выход из игры
    mou.position = (an-360, bn+10)
    mou.click(Button.left, 1)
    # сворачивание окна
    mou.position = (an+130, bn-680)
    mou.click(Button.left, 1)

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
cr_tb = [["LH", 110, 255], ["LS", 100, 255], ["LV", 120, 255], ["UH", 130, 255], ["US", 255, 255], ["UV", 255, 255]]
cr_tb2 = [["LH2", 0, 255], ["LS2", 0, 255], ["LV2", 0, 255], ["UH2", 0, 255], ["US2", 0, 255], ["UV2", 255, 255]]
cr_tb3 = [["LH3", 0, 255], ["LS3", 255, 255], ["LV3", 111, 255], ["UH3", 0, 255], ["US3", 255, 255], ["UV3", 255, 255]]
cr_tb4 = [["LH4", 0, 255], ["LS4", 0, 255], ["LV4", 0, 255], ["UH4", 255, 255], ["US4", 255, 255], ["UV4", 255, 255]]
ctb(cr_tb4)
ctb(cr_tb3)
ctb(cr_tb2)
ctb(cr_tb)
name = ["LH","LS","LV","UH","US","UV"]
name2 = [i+"2" for i in name]
name3 = [i+"3" for i in name]
name4 = [i+"4" for i in name]

left = 3 # 9
wid = 1295 - left # 1294
top = 40 # 40
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
#nn = 0
#m=0

#imggg = cv.imread("screenshoot10.png")

#cv.imshow("bet2")

for i in range(1):
    print("bkjd")
mou = Controller()
an, bn = 1000, 700
rang = 50
mou.position = (460, 1400)
mou.click(Button.left, 1)
#time.sleep(0.3)
#mou.position = (an, bn)
#mou.click(Button.left, 1)
#start(mou, an, bn)
time.sleep(10)

for ina in range(50):
    #if ina == 1:
    #    mou.position = (an, bn)
    #    mou.click(Button.left, 1)
    #if m != 4:
    #    m += 1
    #elif m == 5:
    #    m -= 5
    #    continue

    #ret, frame = cap_new.read()
    #print(type(ret),type(frame))

    img1 = np.array(sct.grab(bbox))
    frame = img1
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #gray = cv.cvtColor(bet, cv.COLOR_BGR2GRAY)

    n = gtbp(name)
    l_b, u_b = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])
    n = gtbp(name2)
    l_b2, u_b2 = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])
    n = gtbp(name3)
    l_b3, u_b3 = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])
    n = gtbp(name4)
    l_b4, u_b4 = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])

    mask = cv.inRange(hsv, l_b, u_b) # charapters
    #res = cv.bitwise_and(frame, frame, mask=mask)
    mask2 = cv.inRange(hsv, l_b2, u_b2) # walls
    #res2 = cv.bitwise_and(frame, frame, mask=mask2)
    mask3 = cv.inRange(hsv, l_b3, u_b3) # boxes
    #res3 = cv.bitwise_and(frame, frame, mask=mask3)
    mask4 = cv.inRange(hsv, l_b4, u_b4) # my circle
    bet_mask = cv.bitwise_or(mask2, mask)
    bet_mask2 = cv.bitwise_or(bet_mask, mask3)
    bet_mask3 = cv.bitwise_or(bet_mask2, mask4)
    #bet = cv.bitwise_or(frame, frame, mask=bet_mask3)

    #bet_con = bet.copy()

    #boxes = b_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in boxes:
    #    cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    #    roi_gray = gray[y:y+h, x:x+w]
    #    roi_color = img1[y:y+h, x:x+w]

    #n_mask = np.zeros_like((mask4.shape[0], mask4.shape[1]))
    n_mask = np.zeros(mask4.shape[:2], dtype = "uint8")
    #print(type(nole), type(mask))
    #imgray = cv.cvtColor(res3, cv.COLOR_BGR2GRAY)
    #rett, thresh = cv.threshold(imgray, 127, 0, 0)
    #contours, _ = cv.findContours(mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(mask4,contours,-1,(0,150,0),3)
    #cv.imshow("bet", mask4)
    #print(contours)

    #RETR_CCOMP или RETR_FLOODFILL
    #bet_cont = cv.bitwise_or(bet, bet, mask=mask3)

    contour, _ = cv.findContours( mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # boxes
    contour2, _ = cv.findContours( mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # charapters
    #contour3, _ = cv.findContours( mask4.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #dc2 = cv.drawContours( bet.copy(), contour, -1, (0, 150, 0), 3)
    #cv2.approxPolyDP()

    for i in contour:
        moments = cv.moments(i, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 500:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)
            cv.circle(bet, (x, y), 10, (255,0,0), -1)
    #cv.circle(bet_con, (bet.shape[1]//2, bet.shape[0]//2), 10, (0,255,0), -1)

    for i in contour2:
        moments = cv.moments(i, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 500:
            x = int(dM10 / dArea)
            y = int(dM01 / dArea)
            cv.circle(n_mask, (x, y+25), 55, (255,255,255), -1)
            cv.circle(bet, (x, y), 10, 1, -1)
            walk(mou, 0, -100, rang, 0.5)

    #print(mask4.size == n_mask.size)
    n_mask = cv.bitwise_and(mask4, n_mask)
    #res_n = cv.bitwise_and(frame, frame, mask=n_mask)
    bet_mask3 = cv.bitwise_or(bet_mask2, n_mask)
    bet = cv.bitwise_or(frame, frame, mask=bet_mask3)

    #cv.namedWindow ( "dc2" , cv.WINDOW_NORMAL)
    #cv.imshow("frame", frame)
    #cv.imshow("mask", n_mask)
    #cv.imshow("bet", bet)
    #cv.imshow("bet1", dc)
    cv.imshow("bet2", bet)
    #cv.imshow("bet3", dc3)
    #cv.imshow("bet4", dc4)

    walk(Controller(), 0, -100, rang, 0.1)

    #if cv.waitKey(3) & 0xFF == ord('4'):
    #    cv.imwrite(f"screenshoot{nn+10}.png", bet)
    #    #cv.imwrite(f"screenshoot{nn+10}.jpg", bet)
    #    nn += 1

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break

print("rjytw")
end(mou, an, bn)
cap.release()