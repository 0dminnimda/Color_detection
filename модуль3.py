import time
import multiprocessing as mp
from multiprocessing import Process
import cv2 as cv
import math as ma

#0,2884559862075313  0,03749471046662279136510921378076  0,00096127574090850863489078621924

def drmon(qq):
    while 1:
        key = qq.get()
        #if key == "c":
        #    qq.put("c")
        #    continue
        if key == "b":
            break
        #else:
        print(key)
        #qq.put("c")

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

def wolk_key(key, dirX, dirY):
    dirX, dirY = map_count(dirX, dirY, 100)

    if dirX == 100 and -100 <= dirY <= 100:
        print(1)
    elif dirY == 100 and -100 <= dirX <= 100:
        print(2)
    elif dirX == -100 and -100 <= dirY <= 100:
        print(3)
    elif dirY == -100 and -100 <= dirX <= 100:
        print(4)

#print(map_count(0, 0, 100))
#from colorama import *
#init() 
#print(Back.YELLOW + 'hh')

jl = "Job location: Melbourn"
text = """
Requisition No: VOI9053459-
Job location: Melbourn
Exp : 2 – 4 Years
Notice period :-15day or less """
import re

job_location=re.compile(r'(?:^Job\s+location:\s+)(.*)$', re.MULTILINE)
#print(job_location.search(text))
job_location2 = job_location.findall(text)[:]#.group()
print(job_location2)
#key_value1 = jl.group(0)
print(re.split(r": |-",jl))

from colorama import init, Fore, Back, Style
init()
while 0:
    print(Style.RESET_ALL+"Чтобы закончить работу, нажмите ENTER, не вводя ничего больше")
    print(Back.BLUE+"Введите значения высоты и ширины, разделяя их пробелом, затем нажмите ENTER\n")
    print(Back.WHITE+Fore.BLACK, end="")
    try:
        inp = input()
        if inp == "":
            break
        inp1, inp2 = inp.split()
        inp1,inp2 = int(inp1),int(inp2)
    except Exception:
        print(Style.RESET_ALL)
        print(Style.BRIGHT+Back.YELLOW+"Попробуйте заново:")
        continue
    print()
    step = 5
    otst = 3
    min1,max1 = 190,270
    min2,max2 = 65,110
    ii1,ii2 = "Не подходит","Не подходит"

    for i in range(min1, max1+1, step):
        if inp1-otst-step < i <= inp1-otst:
            ii1 = i
    else:
        if ii1 != "Не подходит":
            print(Style.RESET_ALL+Fore.GREEN+"Высота коробки -",ii1)
            print(Style.RESET_ALL+Fore.GREEN+"Высота полотна -",ii1-4)
        else:
            print(Style.RESET_ALL+Style.BRIGHT+Fore.RED+"Высота",ii1)

    print()
    for i in range(min2, max2+1, step):
        if inp2-otst-step < i <= inp2-otst:
            ii2 = i
    else:
        if ii2 != "Не подходит":
            print(Style.RESET_ALL+Fore.GREEN+"Ширина коробки -",ii2)
            print(Style.RESET_ALL+Fore.GREEN+"Ширина полотна -",ii2-6.5)
        else:
            print(Style.RESET_ALL+Style.BRIGHT+Fore.RED+"Ширина",ii2)
    print("\n")
    #input('Press ENTER to exit')
#x = -101
a = [1,2,3,4,5,6]
b = [3,4,5,6,7,8]
lambda _:_

#if -100 <= x <= -50:
#    print("dfgxf")

if __name__ == '_main__':
    #st = time.time()
    #for i in range(5):


    qq = mp.Queue()
    qq.get()
    #for _ in range(150):
    pr = Process(target=drmon, args= (qq,), daemon=True)
    pr.start()
    for i in range(15):
        print(i, end = "")
        if i%3 == 0:
            qq.put(i)
            key = i
        else:
            qq.put(key)
        #time.sleep(0.25)
    qq.put("b")
    #pr.join()


    #print("\n",time.time()-st)