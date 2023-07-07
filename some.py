import numpy as np
import cv2 as cv
from mss import mss
import time
import os
from pynput.mouse import Button, Controller as m_c
from pynput.keyboard import Key, Controller as k_c
import math as ma
from random import randint as ra


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
    # выход из игры
    mou.position = (an-360, bn+10)
    mou.click(Button.left, 1)

    # сворачивание окна
    mou.position = (an+130, bn-680)
    mou.click(Button.left, 1)


def create_trackbar_pos(name, val, max=255):
    for i in range(len(name)):
        cv.createTrackbar(name[i], "Tracking", val[i], max, lambda: None)  # nothing


def set_trackbar_pos(name):
    for i, j in name.items():
        cv.setTrackbarPos(i, "Tracking", int(j))


def get_trackbar_pos(name):
    return [cv.getTrackbarPos(i, "Tracking") for i in name]
    

def nothing(x):
    pass


def trackbar_setup(num_of_bars=0, values=None, name="Tracking", def_names=["LH", "LS", "LV", "UH", "US", "UV"]):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    if values is None:
        values = []
    else:
        num_of_bars = len(values)

    names = []

    for i in range(num_of_bars):
        if values is None:
            values += [0 for _ in def_names]

        names += [[f"{name}{i}" for name in def_names]]

        create_trackbar_pos(names[i], values[i])

    # for d in list(zip(values, names)):
    #     print(dict(zip(*d)))

    return names


def trackbar_pos_get(names):
    res = []
    for name in names:
        n = np.array(get_trackbar_pos(name))
        res.append([*np.split(n, 2)])
    return res


def create_masks(hsv, settings):
    masks = []
    for setting in settings:
        masks += [cv.inRange(hsv, *setting)]
        # res = cv.bitwise_and(frame, frame, mask=mask)

    return masks


def betwise(masks, val=0, func=cv.bitwise_or):
    res_mask = np.full_like(np.any(masks), val) # np.all - try, i don't know the results
    for mask in masks:
        res_mask = func(res_mask, mask)

    return res_mask


def main():

    init_values = [
        [0, 0, 200, 0, 0, 255],
        [110, 100, 120, 130, 255, 255],
    ]

    names = trackbar_setup(values=init_values)


    # init_values2 = [
    #     [3, 871, 52, 490]
    # ]

    # names2 = trackbar_setup(values=init_values2, def_names=["left", "wid", "top", "hei"])


    # ["left", "wid", "top", "hei"]
    #cv.namedWindow("Tracking2", cv.WINDOW_NORMAL)
    #cv.createTrackbar("left", "Tracking2", 3, 25, nothing) #left = 3
    #cv.createTrackbar("wid", "Tracking2", 871, 1300, nothing) #wid = 1277 - left
    #cv.createTrackbar("top", "Tracking2", 52, 60, nothing) #top = 42
    #cv.createTrackbar("hei", "Tracking2", 490, 800, nothing) #hei = 759 - top

    left = 3  # cv.getTrackbarPos("left", "Tracking2")
    wid = 871  # cv.getTrackbarPos("wid", "Tracking2")
    top = 52  # cv.getTrackbarPos("top", "Tracking2")
    hei = 490  # cv.getTrackbarPos("hei", "Tracking2")

    sct = mss()
    mou = m_c()
    # key = k_c()

    an, bn = 1000, 700
    rang = 50
    step = 1/16

    while 1:
        img1 = np.array(sct.grab(
            {'top': top, 'left': left, 'width': wid, 'height': hei}))
        frame = img1
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        settings = trackbar_pos_get(names)
        masks = create_masks(hsv, settings)

        general_mask = betwise(masks)

        bet = cv.bitwise_or(frame, frame, mask=general_mask)

        contour4, _ = cv.findContours(mask4.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # "me" yellow

        x0, y0 = wid/2, hei/2
        arr = []
        for i in contour4:
            moments = cv.moments(i, 1)
            dM01 = moments['m01']
            dM10 = moments['m10']
            dArea = moments['m00']

            if dArea > 300:
                x = dM10 / dArea
                y = dM01 / dArea
                x0, y0 = x, y
                cv.circle(bet, (int(x), int(y)), 20, (0, 255, 0), -1)

        clo, dis = closest(arr, wid, hei, x0, y0)

        cv.imshow("bet", bet)
        cv.imshow("mask2", mask2)

        if cv.waitKey(1) & 0xFF == ord('2'):
            break

    # cv.imwrite("some.png", frame)
    sct.shot(output='die_screenshoot.png')
    cv.destroyAllWindows()
    print("end")
    #cap.release()

if __name__ == '__main__':
    main()