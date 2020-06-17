import pandas as pd
import numpy as np
import os
import requests
import shutil

dataset = pd.read_csv("dress_patterns.csv")
x = dataset["category"].unique()

dicty = {}
for i in x:
    os.makedirs('dataset\\'+i)
    dicty[i] = 0

a = dataset["image_url"].values
counter = -1
for i in a:
    index = len(i) - i[::-1].index("/")
    counter += 1
    resp = requests.get(i,stream = True,timeout=None)
    local_file = open('dataset\\'+dataset["category"][counter]+"\\"+i[index:-4],'wb')
    resp.raw.decode_content = True
    shutil.copyfileobj(resp.raw, local_file)
    dicty[dataset["category"][counter]] += 1
    print('dataset\\'+dataset["category"][counter]+"\\"+i[index:-4])
    del resp

    
    
    
    
    
