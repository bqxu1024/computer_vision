import cv2

import numpy as np

from matplotlib import pyplot as plt


# 1.读取图像并转化为灰度图
#首先利用opencv读取图像并转化为灰度图，图像来自于voc2007:
img = cv2.imread("/home/xbq/xbq/computer_vision/data/VOC2007/JPEGImages/000023.jpg")

img = img[:,:,(2,1,0)]
r,g,b = [img[:,:,i] for i in range(3)]
img_gray = r*0.299+g*0.587+b*0.114
plt.imshow(img_gray,cmap="gray")
# 默认使用三通道显示图像。解决方案：在plt.imshow()添加参数
plt.axis('off')
plt.show()



gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow("gray", img)



# 2.显示灰度直方图

# opencv calcHist函数传入5个参数：
# images：图像
# channels：通道
# mask：图像掩码，可以填写None
# hisSize：灰度数目
# ranges：回复分布区间

def histogram(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])
    plt.plot(range(len(hist)), hist)
    plt.show()
histogram(gray)



# 3.直方图均衡化

# 直方图均衡化，这里使用opencv提供的函数：dst
dst = cv2.equalizeHist(gray)

# 均衡化后的图像为：
histogram(dst)

cv2.imshow("histogram", dst)
cv2.waitKey()

