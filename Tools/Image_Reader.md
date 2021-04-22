```python
import base64
import time
import numpy as np

import cv2
import tensorflow as tf
import mxnet as mx
from PIL import Image


class Image_Reader:
    def __init__(self):
        self.sess = tf.Session()

    def read_image_opencv(self, filePath):
        img_data = cv2.imread(filePath)
        return img_data

    def read_image_from_base64(self, filePath):
        with open(filePath, 'r') as f:
            image_code = f.read()
        img_data = self.base64_cv2(image_code)
        return img_data

    def base64_cv2(self, base64_str):
        imgString = base64.b64decode(base64_str)
        nparr = np.fromstring(imgString, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    def cv2_base64(self, image):
        base64_str = cv2.imencode('.jpg', image)[1].tostring()
        base64_str = base64.b64encode(base64_str)
        return base64_str

    def read_image_PIL(self, filePath):
        img_data = Image.open(filePath)
        img_data = np.asarray(img_data)
        return img_data

    def read_image_MX(self, filePath):
        img_data = mx.image.imdecode(open(filePath, 'rb').read())
        return img_data.asnumpy()

    def read_image_TF(self, filePath):
        img = tf.gfile.FastGFile(filePath, 'rb').read()
        img_data = tf.image.decode_jpeg(img)
        img_data = img_data.eval(session=self.sess)
        return img_data


if __name__ == '__main__':
    # imagePath = "test1_1080_2396.jpg"  
    imagePath = "test2_640_480.jpg"
    image_base64_path = "image.txt"
    img_reader = Image_Reader()

    # opencv 读取图片
    start = time.time()
    for i in range(10):
        img_data = img_reader.read_image_opencv(imagePath)
    end = time.time()
    print ("opencv cost: ", (end-start)/10*1000, "ms")

    # base64编码 读取图片
    start = time.time()
    for i in range(10):
        img_data = img_reader.read_image_from_base64(image_base64_path)
    end = time.time()
    print("base64 cost: ", (end-start)/10*1000, "ms")

    # PIL 读取图片
    start = time.time()
    for i in range(10):
        img_data = img_reader.read_image_PIL(imagePath)
    end = time.time()
    print("PIL cost: ", (end-start)/10*1000, "ms")

    # Tensorflow 读取图片
    start = time.time()
    for i in range(10):
        img_data = img_reader.read_image_TF(imagePath)
    end = time.time()
    print("Tensorflow cost: ", (end-start)/10*1000, "ms")

    # mxnet 读取图片
    start = time.time()
    for i in range(10):
        img_data = img_reader.read_image_MX(imagePath)
    end = time.time()
    print("mxnet cost: ", (end-start)/10*1000, "ms")
```

# 测试结果：  
1920 * 1080 分辨率  （HDTV 1080i）  
【opencv】：45.54038047790527 ms  
【base64】：54.015493392944336 ms  
【PIL】：27.900218963623047 ms  
【Tensorflow】：87.90326118469238 ms  
【mxnet】：93.0964469909668 ms

1366 * 768 分辨率 （HDMI）  
【opencv】：22.500014305114746 ms  
【base64】：56.99951648712158 ms  
【PIL】：12.504076957702637 ms  
【Tensorflow】：51.57310962677002 ms  
【mxnet】：63.30103874206543 ms

1280 * 720 分辨率  （HDTV 720p）  
【opencv】：19.80435848236084 ms  
【base64】：61.210083961486816 ms  
【PIL】：15.901708602905275 ms  
【Tensorflow】：57.00078010559082 ms  
【mxnet】：54.51693534851074 ms  

1080 * 2396 分辨率   
【opencv】：45.20227909088135 ms  
【base64】：56.400203704833984 ms  
【PIL】：32.40470886230469 ms  
【Tensorflow】：100.76427459716797 ms  
【mxnet】：89.70215320587158 ms  

704 * 480 分辨率 （EDTV 480p）  
【opencv】：4.700112342834473 ms  
【base64】：42.70508289337158 ms  
【PIL】：4.09998893737793 ms  
【Tensorflow】：10.600137710571289 ms  
【mxnet】：32.50222206115723 ms  

640 * 480 分辨率   
【opencv】：9.803581237792969 ms  
【base64】：44.596290588378906 ms  
【PIL】：8.005261421203613 ms  
【Tensorflow】：20.516490936279297 ms  
【mxnet】：45.599961280822754 ms  

250 * 250 分辨率  
【opencv】：1.1983633041381836 ms  
【base64】：47.60618209838867 ms  
【PIL】：2.003002166748047 ms  
【Tensorflow】：7.396864891052246 ms  
【mxnet】：25.00326633453369 ms  

- 对于常见分辨率，读取速度普遍较快（几毫秒），其中 opencv 的读取速度较快
- 对于非常见分辨率，读取速度普遍较慢（几十毫秒），其中 PIL 的读取速度较快


