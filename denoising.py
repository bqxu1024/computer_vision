import cv2
import numpy as np
import skimage
from skimage.util.dtype import convert

# 1.读取图像
img = cv2.imread("/home/xbq/xbq/computer_vision/data/VOC2007/JPEGImages/000032.jpg")
cv2.imshow("origin_img",img)
cv2.waitKey()




# 2.添加噪声
# # # 方法1：用第三方工具添加噪声
# noise_img = skimage.util.random_noise(img, mode="gaussian")
# # 就用Python第三方库scikit-image的random_noise添加噪声：
# # mode是可选参数：分别有'gaussian'、'localvar'、'salt'、'pepper'、's&p'、'speckle'，可以选择添加不同的噪声类型。


# # 方法2：用numpy生成噪声
def add_noise(img):
    img = np.multiply(img, 1. / 255,
                        dtype=np.float64)
    mean, var = 0, 0.01
    noise = np.random.normal(mean, var ** 0.5,
                             img.shape)
    img = convert(img, np.floating)
    out = img + noise
    return out
noise_img = add_noise(img)
gray_img =  cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)


cv2.imshow("noise_img", noise_img)
cv2.waitKey()




# 3.图像去噪
# 方法1：用第三方工具去噪: 使用opencv这一类工具进行去噪
denoise = cv2.medianBlur(img, ksize=3)   # 中值滤波
# denoise = cv2.fastNlMeansDenoising(img, ksize=3)  # 均值滤波
# denoise = cv2.GaussianBlur(img, ksize=3)    # 高斯滤波


# # 方法2： 编程一步一步实现图像去噪，首先是计算窗口邻域内的值，这里以计算中值为例：
# def compute_pixel_value(img, i, j, ksize, channel):
#     h_begin = max(0, i - ksize // 2)
#     h_end = min(img.shape[0], i + ksize // 2)
#     w_begin = max(0, j - ksize // 2)
#     w_end = min(img.shape[1], j + ksize // 2)
#     return np.median(img[h_begin:h_end, w_begin:w_end, channel])
#
# #  方法2：去噪   对每个像素使用compute_pixel_value函数计算新像素的值：
# def denoise(img, ksize):
#     output = np.zeros(img.shape)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             output[i, j, 0] = compute_pixel_value(img, i, j, ksize, 0)
#             output[i, j, 1] = compute_pixel_value(img, i, j, ksize, 1)
#             output[i, j, 2] = compute_pixel_value(img, i, j, ksize, 2)
#     return output

output = denoise(noise_img, 3)

cv2.imshow("denoise_img", noise_img)
cv2.waitKey()


