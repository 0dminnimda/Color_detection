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
#start = time.time()
nn = 0
m=0

imggg = cv.imread("screenshoot10.png")

while 1:
    #if m != 4:
    #    m += 1
    #elif m == 5:
    #    m -= 5
    #    continue

    #ret, frame = cap_new.read()
    #print(type(ret),type(frame))

    sct_img = sct.grab(bbox)
    img1 = np.array(sct_img)
    frame = img1
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #gray = cv.cvtColor(bet, cv.COLOR_BGR2GRAY)

    mask = cv.inRange(hsv, l_b, u_b)
    #res = cv.bitwise_and(frame, frame, mask=mask)
    mask2 = cv.inRange(hsv, l_b2, u_b2)
    #res2 = cv.bitwise_and(frame, frame, mask=mask2)
    mask3 = cv.inRange(hsv, l_b3, u_b3)
    #res3 = cv.bitwise_and(frame, frame, mask=mask3)
    bet_mask = cv.bitwise_or(mask2, mask)
    bet_mask2 = cv.bitwise_or(bet_mask, mask3)
    bet = cv.bitwise_or(frame, frame, mask=bet_mask2)

    bet_con = bet.copy()

    #boxes = b_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in boxes:
    #    cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    #    roi_gray = gray[y:y+h, x:x+w]
    #    roi_color = img1[y:y+h, x:x+w]

    nole = np.zeros((bet.shape[0], bet.shape[1]))
    #imgray = cv.cvtColor(res3, cv.COLOR_BGR2GRAY)
    #rett, thresh = cv.threshold(imgray, 127, 0, 0)
    #contours, _ = cv.findContours(mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #cv.drawContours(mask4,contours,-1,(0,150,0),3)
    #cv.imshow("bet", mask4)
    #print(contours)

    #RETR_CCOMP или RETR_FLOODFILL
    #bet_cont = cv.bitwise_or(bet, bet, mask=mask3)

    contour, _ = cv.findContours( mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # внешние контуры
    #a3, fc4 = cv.findContours( mask3.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # многоуровневая иерархия

    #dc2 = cv.drawContours( bet.copy(), contour, -1, (0, 150, 0), 3)

    #cv2.approxPolyDP()

    for i in contour:
        cv.drawContours( bet_con, i, -1, (0, 150, 0), 3)
        #p;cv.drawContours( nole, i, -1, (0, 150, 0), 3)
        #cv.imshow("bet", nole)
        #print(i)
    #moments = cv.moments(mask3.copy(), 1)
    #dM01 = moments['m01']
    #dM10 = moments['m10']
    #dArea = moments['m00']

    #if dArea > 150:
    #    x = int(dM10 / dArea)
    #    y = int(dM01 / dArea)
    #    cv.circle(dc2, (x, y), 10, (255,0,255), -1)

    #cv.namedWindow ( "dc2" , cv.WINDOW_NORMAL)
    #cv.imshow("frame", frame)
    #cv.imshow("bet", bet)
    #cv.imshow("bet1", dc)
    cv.imshow("bet2", bet_con)
    #cv.imshow("bet3", dc3)
    #cv.imshow("bet4", dc4)

    if cv.waitKey(3) & 0xFF == ord('4'):
        cv.imwrite(f"screenshoot{nn+10}.png", bet)
        #cv.imwrite(f"screenshoot{nn+10}.jpg", bet)
        nn += 1

    if cv.waitKey(1) & 0xFF == ord('2'):
        cv.destroyAllWindows()
        break

cap.release()