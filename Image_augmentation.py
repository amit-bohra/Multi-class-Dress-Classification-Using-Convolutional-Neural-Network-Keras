import cv2
import os
import pandas as pd
import numpy as np
import random as rd


def flip(img):
    return cv2.flip(img,1)

dataset = pd.read_csv("dress_patterns.csv")
values = dataset["category"].unique()
for i in values:
    flag = 1
    files = os.listdir("dataset\\"+i)
    while len(files) < 1000:
        files = os.listdir("dataset\\"+i)
        rd.shuffle(files)
        diff = 1000-len(files)
        at_max = (diff//6)+1
        counter = 0
        for j in files:
            counter += 1
            frame = cv2.imread("dataset\\"+i+"\\"+j)
            if flag == 1:
                frame = cv2.flip(frame,1)
            elif flag == 2:
                listy = [0.2,0.5,2.2,2.5,2.7]
                x = rd.choice(listy)
                invGamma = 1.0 / x
                table = [((i/255.0)**invGamma)*255 for i in range(0,256)]
                table = np.array(table)
                table = np.uint8(table)
                frame = cv2.LUT(frame, table)
            elif flag == 3:
                for channel in range(frame.shape[2]):
                    frame[:,:,channel] = cv2.equalizeHist(frame[:,:,channel])
            elif flag == 4:
                r,c,l=frame.shape
                tx= rd.choice(list(range(40, 90)))
                ty= rd.choice(list(range(40, 90)))
                options = [1,2]
                chosen = rd.choice(options)
                if chosen == 2:
                    tx = -tx
                    ty = -ty
                p=np.float32([[1,0,tx],[0,1,ty]])
                frame=cv2.warpAffine(frame,p,(c,r))
            elif flag == 5:
                frame = frame[:,:,::-1]
            elif flag == 6:
                pts1=np.float32([[110,110],[210,110],[110,210],[210,210]])
                x= rd.choice(list(range(15,30)))
                y= rd.choice(list(range(15,30)))
                pts2=np.float32([[110-x,110-y],[210+x,110-y],[110-x,210+y],[210+x,210+y]])
                M=cv2.getPerspectiveTransform(pts1,pts2)
                frame=cv2.warpPerspective(frame,M,(frame.shape[1],frame.shape[0]))
            cv2.imwrite("dataset\\"+i+"\\"+str(flag)+"_"+j,frame)
        flag += 1

dicty1 = {}
for i in values:
    files = os.listdir("dataset\\"+i)
    dicty1[i] = len(files)
print(dicty1)


dicty = {}
for i in values:
    os.makedirs('new_dataset\\training_set\\'+i)
    os.makedirs('new_dataset\\test_set\\'+i)
    dicty[i] = 0
    
for i in values:
    files = os.listdir("dataset\\"+i)
    rd.shuffle(files)
    new_files = files[:200]
    remaining = files[200:]
    for j in new_files:
        frame = cv2.imread("dataset\\"+i+"\\"+j)
        cv2.imwrite("new_dataset\\test_set\\"+i+"\\"+j,frame)
    sample = rd.sample(remaining,1800)
    for j in sample:
        frame = cv2.imread("dataset\\"+i+"\\"+j)
        cv2.imwrite("new_dataset\\training_set\\"+i+"\\"+j,frame)
print('done')

dicty2 = {}
for i in values:
    files = os.listdir('new_dataset\\training_set\\'+i)
    dicty2[i] = len(files)

dicty3 = {}
for i in values:
    files = os.listdir('new_dataset\\test_set\\'+i)
    dicty3[i] = len(files)

print(dicty2)
print()
print(dicty3)
                
