import os
import numpy as np

xml_dir = 'D:/SignData/Annotations'
imageSetsList_dir = 'D:/SignData/ImageSets/Main/'

imgList = []

for file_name in os.listdir(xml_dir):
    imgList.append(file_name.split('.')[0])
imgList = np.array(imgList)
index = np.arange(0, len(imgList), 1)
np.random.shuffle(index)
index = index.astype(dtype=int)

trainList = imgList[index[0:3000]]
print(trainList.shape[0])
valList = imgList[index[3000:]]

ftxt = open(imageSetsList_dir+'sign_train.txt', 'w')
for t in trainList:
    ftxt.write(t+'\n')
ftxt.close()

ftxt = open(imageSetsList_dir+'sign_val.txt','w')
for v in valList:
    ftxt.write(v+'\n')
ftxt.close()
