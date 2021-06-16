import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../outputs/output_44148.jpg0.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
dst = cv2.filter2D(img,-1,kernel)
gaussian = cv2.GaussianBlur(img,(3,3),0)

norm_img = np.zeros((200,200))
normal = cv2.normalize(img,norm_img,0,255,norm_type=cv2.NORM_MINMAX)
normal = np.divide(normal,255)

cv2.imshow('s',img)
image = [img,dst,gaussian,normal]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(image[i])
    plt.xticks([]),plt.yticks([])

plt.show()