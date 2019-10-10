import numpy as np
import scipy as sp
import cv2 as cv
from mss import mss
import time
from pynput import mouse
import os
#from pynput import keyboard
from pynput.mouse import Button, Controller
import math as ma
import random
#from mose1 import map_count, shoot, walk, end

ra = random.randint

def nothing(x):
    pass
    pass

def map_count(ab1, ac1, n, nn):
    #ab1, ac1 = input().split()
    #ab1, ac1 = int(ab1), int(ac1)
    if nn == True:
        ab1, ac1 = (-1292+ab1), (-730+ac1)
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
    mou.position = (460, 1425)
    mou.click(Button.left, 1)

    time.sleep(5)
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

def closest(arr, X, Y):
    if arr == []:
        img = np.zeros((Y, X, 3), dtype = "uint8")
        cv.imshow("img", img)
        return False
    else:
        img = np.zeros((Y, X, 3), dtype = "uint8")
        m = arr[0]
        for j in arr:
            cv.circle(img, (int(j[0]+X/2), int(j[1]+Y/2)), 15, (255,0,0), 2)
            cv.putText(img, "%d" % (ma.sqrt(ma.fabs(j[0])**2 + ma.fabs(j[1])**2)), (int(j[0]+X/2)+10,int(j[1]+Y/2)-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            new = ma.sqrt(ma.fabs(j[0])**2 + ma.fabs(j[1])**2)
            old = ma.sqrt(ma.fabs(m[0])**2 + ma.fabs(m[1])**2)
            if new < old:
                m = j
        cv.circle(img, (int(m[0]+X/2), int(m[1]+Y/2)), 15, (0,0,255), -1)
        cv.putText(img, "%d" % (ma.sqrt(ma.fabs(m[0])**2 + ma.fabs(m[1])**2)), (int(m[0]+X/2)+10,int(m[1]+Y/2)-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv.imshow("img", img)
        return m

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

for _ in range(1):
    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
    cr_tb = [["LH", 110, 255], ["LS", 100, 255], ["LV", 120, 255], ["UH", 130, 255], ["US", 255, 255], ["UV", 255, 255]]
    cr_tb2 = [["LH2", 0, 255], ["LS2", 0, 255], ["LV2", 0, 255], ["UH2", 0, 255], ["US2", 0, 255], ["UV2", 255, 255]]
    cr_tb3 = [["LH3", 0, 255], ["LS3", 255, 255], ["LV3", 111, 255], ["UH3", 0, 255], ["US3", 255, 255], ["UV3", 255, 255]]
    cr_tb4 = [["LH4", 30, 255], ["LS4", 158, 255], ["LV4", 67, 255], ["UH4", 62, 255], ["US4", 200, 255], ["UV4", 255, 255]]
    ctb(cr_tb4)
    ctb(cr_tb3)
    ctb(cr_tb2)
    ctb(cr_tb)
    name = ["LH","LS","LV","UH","US","UV"]
    name2 = [i+"2" for i in name]
    name3 = [i+"3" for i in name]
    name4 = [i+"4" for i in name]

    cap = cv.VideoCapture('2019-10-06 02-16-00.mp4')
    cap2 = cv.VideoCapture('2019-10-06 02-17-30.mp4')
    cap_new = cv.VideoCapture('2019-10-07 12-58-00.mp4')#2019-10-07 12-45-36.mp4')#2019-10-07 12-06-37.mp4')
    cap_new.set(cv.CAP_PROP_FPS, 5)
    #hei = cap_new.get(cv.CAP_PROP_FRAME_HEIGHT)
    #wid = cap_new.get(cv.CAP_PROP_FRAME_WIDTH)
    #cap_new.set(cv.CAP_PROP_FRAME_WIDTH , 1024)
    #cap_new.set(cv.CAP_PROP_FRAME_HEIGHT , 720)

    imggg = cv.imread("Brawl Stars_Screenshot_2019.10.10_21.59.28.jpg")
    pass

left = 3 # 9
wid = 1295 - left # 1294
top = 30 # 40
hei = 770 - top

bbox = {'top': top, 'left': left, 'width': wid, 'height': hei}
sct = mss()

mou = Controller()
an, bn = 1000, 700
rang = 50
#start(mou, an, bn)
step = 0.3
#time.sleep(8)

#for ina in range(75):
while 1:
    #if m != 4:
    #    m += 1
    #elif m == 5:
    #    m -= 5
    #    continue

    #ret, frame = cap_new.read()

    img1 = np.array(sct.grab(bbox))
    frame = img1
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #gray = cv.cvtColor(bet, cv.COLOR_BGR2GRAY)

    for _ in range(1):
        n = gtbp(name)
        l_b, u_b = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])
        n = gtbp(name2)
        l_b2, u_b2 = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])
        n = gtbp(name3)
        l_b3, u_b3 = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])
        n = gtbp(name4)
        l_b4, u_b4 = np.array([n[0], n[1], n[2]]), np.array([n[3], n[4], n[5]])

        mask = cv.inRange(hsv, l_b, u_b) # charapters
        res = cv.bitwise_and(frame, frame, mask=mask)
        mask2 = cv.inRange(hsv, l_b2, u_b2) # walls
        res2 = cv.bitwise_and(frame, frame, mask=mask2)
        mask3 = cv.inRange(hsv, l_b3, u_b3) # boxes
        res3 = cv.bitwise_and(frame, frame, mask=mask3)
        mask4 = cv.inRange(hsv, l_b4, u_b4) # my circle
        res4 = cv.bitwise_and(frame, frame, mask=mask4)
        bet_mask = cv.bitwise_or(mask2, mask)
        bet_mask2 = cv.bitwise_or(bet_mask, mask3)
        bet_mask3 = cv.bitwise_or(bet_mask2, mask4)
        #n_mask = cv.bitwise_and(mask4, n_mask)
        #res_n = cv.bitwise_and(frame, frame, mask=n_mask)
        #bet_mask3 = cv.bitwise_or(bet_mask2, n_mask)
        bet = cv.bitwise_or(frame, frame, mask=bet_mask3)
        #bet = cv.bitwise_or(frame, frame, mask=bet_mask3)
        pass

    #n_mask = np.zeros_like((mask4.shape[0], mask4.shape[1]))
    #n_mask = np.zeros(mask4.shape[:2], dtype = "uint8")
    #points_n = np.zeros(frame.shape[:2], dtype = "uint8")

    #RETR_CCOMP или RETR_FLOODFILL
    #bet_cont = cv.bitwise_or(bet, bet, mask=mask3)

    contour, _ = cv.findContours( mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # boxes
    contour2, _ = cv.findContours( mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # charapters
    #contour3, _ = cv.findContours( mask4.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv2.approxPolyDP()

    arr = []
    for i in contour:
        moments = cv.moments(i, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 500:
            x = dM10 / dArea
            y = dM01 / dArea
            cv.circle(bet, (int(x), int(y)), 10, (255,0,0), -1)
            arr.append([x-(wid/2),y-(hei/2)])

    clo = closest(arr, wid, hei)
    if clo != False:
        pass#walk(mou, clo[0], clo[1], rang, step)#, True)

    for i in contour2:
        moments = cv.moments(i, 1)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea > 500:
            x = dM10 / dArea
            y = dM01 / dArea
            #cv.circle(n_mask, (x-5, y+25), 60, (255,255,255), -1)
            cv.circle(bet, (int(x), int(y)), 10, (0,255,0), -1)
            #cv.imshow("bet", points)
            
    #points = cv.bitwise_and(points, points, mask = points)
    #print(mask4.size == n_mask.size)
    #print(frame.dtype)
    #bet = cv.bitwise_and(bet, points)

    #cv.namedWindow ( "орлоры" , cv.WINDOW_NORMAL)
    #cv.imshow("frame", frame)
    #cv.imshow("hsv", hsv)
    #cv.imshow("res", res)
    #cv.imshow("res2", res2)
    #cv.imshow("res3", res3)
    #cv.imshow("res4", res4)
    #cv.imshow("bet", bet)

    #if cv.waitKey(3) & 0xFF == ord('4'):
    #    cv.imwrite(f"screenshoot{nn+10}.png", bet)
    #    #cv.imwrite(f"screenshoot{nn+10}.jpg", bet)
    #    nn += 1

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break

print("end")
#end(mou, an, bn)
cap.release()