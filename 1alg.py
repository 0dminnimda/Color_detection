import numpy as np
import scipy as sp
import cv2 as cv
from mss import mss
import time
import os
from pynput.mouse import Button, Controller as m_c
from pynput.keyboard import Key, Controller as k_c
import math as ma
from random import randint as ra
import multiprocessing as mp
from multiprocessing import Process

def nothing(x):
    pass
    pass

def map_count(ab1, ac1, n, nn = False):
    if nn == True:
        ab1, ac1 = (-1274+ab1), (-717+ac1)
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

def shoot(mou, dirX, dirY, rangee):
    mou.position = (1200, 650)
    t = 0.1**1.125
    dirX, dirY = map_count(dirX, dirY, rangee)
    mou.press(Button.left)
    time.sleep(t/1.25)
    #mou.position = (1200+dirX, 650+dirY)
    mou.move(dirX, dirY)
    time.sleep(t/1.25)
    mou.release(Button.left)
    pass

def walk(mou, dirX, dirY, rangee, tim):
    mou.position = (150, 660)
    t = 0.1**1.125
    dirX, dirY = map_count(dirX, dirY, rangee)
    mou.press(Button.left)
    time.sleep(t/2)
    #mou.position = (1200+dirX, 650+dirY)
    mou.move(dirX, dirY)
    time.sleep(tim-t/2)
    mou.release(Button.left)

def k_prel(lit, key, ti):
    key.press(lit)
    time.sleep(ti)
    key.release(lit)
    return lit

def k_dou_prel(lit1, lit2, key, ti):
    key.press(lit1)
    key.press(lit2)
    time.sleep(ti)
    key.release(lit1)
    key.release(lit2)
    return lit1, lit2

def walk_key(key, dirX, dirY):

    n1 = 25
    n2 = n1 + n1*ma.sqrt(2)
    an = n2/3

    t = 0.1
    val = None

    x, y = map_count(dirX, dirY, n2)
    y = -y
    dirX, dirY = ma.fabs(x), ma.fabs(y)

    #print(x, y)
    if ma.fabs(x) > ma.fabs(y):
        if x > 0:
            if 25 > y > -25:
                val = k_prel('в', key, t)
            elif -25 > y > -n2-1:
                val = k_dou_prel('ы', 'в', key, t)
            elif 25 < y < n2+1:
                val = k_dou_prel('в', 'ц', key, t)
        if x < 0:
            if 25 > y > -25:
                val = k_prel('ф', key, t)
            elif -25 > y > -n2-1:
                val = k_dou_prel('ы', 'ф', key, t)
            elif 25 < y < n2+1:
                val = k_dou_prel('ф', 'ц', key, t)

    elif ma.fabs(x) < ma.fabs(y):
        if y > 0:
            if 25 > x > -25:
                val = k_prel('ц', key, t)
            elif -25 > x > -n2-1:
                val = k_dou_prel('ф', 'ц', key, t)
            elif 25 < x < n2+1:
                val = k_dou_prel('ц', 'в', key, t)
        if y < 0:
            if 25 > x > -25:
                val = k_prel('ы', key, t)
            elif -25 > x > -n2-1:
                val = k_dou_prel('ф', 'ы', key, t)
            elif 25 < x < n2+1:
                val = k_dou_prel('ы', 'в', key, t)

    else:
        raise RuntimeError

    #print(x, y, val)
    pass

def w_call(qq):
    ke = k_c()
    while 1:
        key = qq.get()
        walk_key(ke,*key)
        pass

def s_call(qq):
    while 1:
        key = qq.get()
        shoot(*key)
        pass

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

def closest(arr, X, Y, x0, y0):
    r = 2
    if arr == []:
        img = np.zeros((Y, X, 3), dtype = "uint8")
        cv.line(img, (0, int(y0)), (X, int(y0)), (255, 0, 0), 2) # hor
        cv.line(img, (int(x0), 0), (int(x0), Y), (255, 0, 0), 2) # ver
        cv.line(img, (int( X/2 + X/(2*(1 + ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(0 - Y/2+y0)), (int( X/2 - X/(2*(1+ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(Y - Y/2+y0)), (255, 0, 0), 2)
        cv.line(img, (int( X/2 - X/(2*(1 + ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(0 - Y/2+y0)), (int( X/2 + X/(2*(1+ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(Y - Y/2+y0)), (255, 0, 0), 2)
        cv.circle(img, (int(x0), int(y0)), 10, (0,0,255), 3)
        cv.putText(img, "0", (int(x0)-10, int(y0)+40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv.imshow("img", img)
        return False, False
    else:
        img = np.zeros((Y, X, 3), dtype = "uint8")
        m = arr[0]
        c = 0
        for j in arr:
            c += 1
            new = ma.sqrt(ma.fabs(j[0]-x0)**2 + ma.fabs(j[1]-y0)**2)
            old = ma.sqrt(ma.fabs(m[0]-x0)**2 + ma.fabs(m[1]-y0)**2)
            cv.circle(img, (int(j[0]), int(j[1])), 15, (255,0,0), 2)
            cv.putText(img, "%d" % new, (int(j[0])+10,int(j[1])-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if new < old:
                m = j
        cv.line(img, (0, int(y0)), (X, int(y0)), (255, 0, 0), 2) # hor
        cv.line(img, (int(x0), 0), (int(x0), Y), (255, 0, 0), 2) # ver
        cv.line(img, (int( X/2 + X/(2*(1 + ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(0 - Y/2+y0)), (int( X/2 - X/(2*(1+ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(Y - Y/2+y0)), (255, 0, 0), 2)
        cv.line(img, (int( X/2 - X/(2*(1 + ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(0 - Y/2+y0)), (int( X/2 + X/(2*(1+ma.sqrt(2)))*ma.sqrt(r) - X/2+x0 ), int(Y - Y/2+y0)), (255, 0, 0), 2)
        cv.line(img, (int(x0), int(y0)), (int(m[0]), int(m[1])), (0, 0, 255), 2)
        cv.circle(img, (int(m[0]), int(m[1])), 15, (0, 0, 255), -1)
        cv.circle(img, (int(x0), int(y0)), 10, (255, 0, 0), 3)
        cv.putText(img, "%d" % c, (int(x0)-10, int(y0)+40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv.imshow("img", img)
        m[0], m[1] = m[0]-x0, m[1]-y0
        return m, old

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

def main_f():
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

    qq = mp.Queue()
    qq_2 = mp.Queue()
    #qq_3 = mp.Queue()
    #qq2 = mp.Queue()
    pr = Process(target=w_call, args=(qq,), daemon=True)
    pr_2 = Process(target=w_call, args=(qq_2,), daemon=True)
    #pr_3 = Process(target=w_call, args=(qq_3,), daemon=True)
    #pr2 = Process(target=s_call, args=(qq2,), daemon=True)
    val = 2 
    c, c2 = 0, val//2

    left = 3
    wid = 1277 - left
    top = 42
    hei = 759 - top

    sct = mss()
    mou = m_c()
    #key = k_c()

    an, bn = 1000, 700
    rang = 50
    step = 1/16

    #start(mou, an, bn)
    #time.sleep(8)
    pr.start()
    pr_2.start()
    #pr_3.start()
    #pr2.start()

    #for _ in range(400):
    #st = time.time()
    while 1:
        #if m != 4:
        #    m += 1
        #elif m == 5:
        #    m -= 5
        #    continue

        #ret, frame = cap_new.read()

        img1 = np.array(sct.grab({'top': top, 'left': left, 'width': wid, 'height': hei}))
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
            #res = cv.bitwise_and(frame, frame, mask=mask)
            mask2 = cv.inRange(hsv, l_b2, u_b2) # walls
            #res2 = cv.bitwise_and(frame, frame, mask=mask2)
            mask3 = cv.inRange(hsv, l_b3, u_b3) # boxes
            #res3 = cv.bitwise_and(frame, frame, mask=mask3)
            mask4 = cv.inRange(hsv, l_b4, u_b4) # my circle
            res4 = cv.bitwise_and(frame, frame, mask=mask4)
            bet_mask = cv.bitwise_or(mask2, mask)
            bet_mask2 = cv.bitwise_or(bet_mask, mask3)
            bet_mask3 = cv.bitwise_or(bet_mask2, mask4)
            bet = cv.bitwise_or(frame, frame, mask=bet_mask3)
            pass

        #points_n = np.zeros(frame.shape[:2], dtype = "uint8")

        #cv2.approxPolyDP()
        #RETR_CCOMP или RETR_FLOODFILL
        #bet_cont = cv.bitwise_or(bet, bet, mask=mask3)

        contour, _ = cv.findContours( mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # boxes
        contour2, _ = cv.findContours( mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # charapters
        contour3, _ = cv.findContours( mask4.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # "me"

        arrx, arry = [], []
        for i in contour3:
            moments = cv.moments(i, 1)
            dM01 = moments['m01']
            dM10 = moments['m10']
            dArea = moments['m00']

            if dArea > 100:
                x = dM10 / dArea
                y = dM01 / dArea
                #cv.circle(bet, (int(x), int(y)), 10, (0,255,255), -1)
                arrx.append(x)
                arry.append(y)

        if arrx != []:
            mea = (np.mean(arrx), np.mean(arry))
            cv.circle(bet, (int(mea[0]), int(mea[1])), 10, (0,255,255), -1)
            #print((mea[0], mea[1]),(mea[0]+50, mea[1]-50))
            #cv.rectangle(bet, (int(mea[0]-37.5), int(mea[1])), (int(mea[0]+37.5), int(mea[1])-60), (0,255,255))
            pass
        else:
           mea = []
    
        x0, y0 = wid/2, hei/2
        for i in contour2:
            moments = cv.moments(i, 1)
            dM01 = moments['m01']
            dM10 = moments['m10']
            dArea = moments['m00']

            if dArea > 300:
                x = dM10 / dArea
                y = dM01 / dArea
                cv.circle(bet, (int(x), int(y)), 10, (0,255,0), -1)
                #cv.circle(n_mask, (x-5, y+25), 60, (255,255,255), -1)
                #cv.imshow("bet", points)

                if mea != [] and mea[0]-35 < x < mea[0]+35 and mea[1]-60 < y < mea[1]+15:
                    x0, y0 = x, y
                    cv.circle(bet, (int(x), int(y)), 20, (0,255,0), -1)

        arr = []
        for i in contour:
            moments = cv.moments(i, 1)
            dM01 = moments['m01']
            dM10 = moments['m10']
            dArea = moments['m00']

            if dArea > 200:
                x = dM10 / dArea
                y = dM01 / dArea
                cv.circle(bet, (int(x), int(y)), 10, (255,0,0), -1)
                arr.append([x,y])

        clo, dis = closest(arr, wid, hei, x0, y0)

        if dis < 10:
            pass#break

        c += 1
        c2 += 1
        #c3 += 1
        if clo != False:
            if c % val == 0:
                c -= val
                qq.put((clo[0], clo[1]))
                #qq_key = (clo[0], clo[1])

            #if c2 % val == 0:
            #    c2 -= val
            #    qq_2.put((clo[0], clo[1]))

            #if c3 % val == 0:
            #    c3 -= val
            #    qq_3.put((clo[0], clo[1]))
                #qq.put((mou, clo[0], clo[1], rang, step))
                #qq_key = (mou, clo[0], clo[1], rang, step)
                #qq2.put((mou, clo[0], clo[1], rang))
                #qq_key2 = (mou, clo[0], clo[1], rang)
            #else:
                #pass
                #qq.put(qq_key)
                #qq2.put(qq_key2)

        #cv.namedWindow("орлоры", cv.WINDOW_NORMAL)
        #cv.imshow("frame", frame)
        #cv.imshow("hsv", hsv)
        #cv.imshow("res", res)
        #cv.imshow("res2", res2)
        #cv.imshow("res3", res3)
        #cv.imshow("res4", res4)
        cv.imshow("bet", bet)

        #if cv.waitKey(3) & 0xFF == ord('4'):
        #    cv.imwrite(f"screenshoot-t0.png", bet)
        #    #cv.imwrite(f"screenshoot{nn+10}.jpg", bet)
        #    #nn += 1

        if cv.waitKey(1) & 0xFF == ord('2'):
            #print(c/(time.time()-st))
            #print((time.time()-st)/c)
            break
    
    sct.shot(output='die_screenshoot.png')
    pr.terminate()
    pr_2.terminate()
    #pr_3.terminate()
    cv.destroyAllWindows()
    print("end")
    #end(mou, an, bn)
    cap.release()

if __name__ == '__main__':

    main_f()