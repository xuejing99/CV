import cv2
import numpy as np


# 将 NV21 的裸数据图片格式转换为 RGB 图片，并保存
def NV212RGB(yuv_path, width, height, save_path):
    with open(yuv_path, 'rb') as f:
        yuvdata = np.fromfile(f, dtype=np.uint8)
    cv_format=cv2.COLOR_YUV2BGR_NV21
    bgr_img = cv2.cvtColor(yuvdata.reshape((height*3//2, width)), cv_format)  
    cv2.imwrite(save_path, bgr_img)
    return bgr_img
  
  
if __name__ == '__main__':
    # 将 NV21 的裸数据图片格式转换为 RGB 图片，并保存
    # 需指定图片的 宽、高
    yuv_path, save_path = "640x480_1.NV21", "1.jpg"
    width, height = 640, 480
    bgr_img = NV212RGB(yuv_path, width, height, save_path)
