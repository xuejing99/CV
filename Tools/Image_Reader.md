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
【opencv】：3.399944305419922 ms  
【base64】：51.909708976745605 ms  
【PIL】：3.9001941680908203 ms  
【Tensorflow】：8.99970531463623 ms  
【mxnet】：33.109235763549805 ms

1366 * 768 分辨率 （HDMI）  
【opencv】：1.7000675201416016 ms  
【base64】：63.70885372161865 ms  
【PIL】：2.899765968322754 ms  
【Tensorflow】：7.400035858154297 ms  
【mxnet】：27.506351470947266 ms

1280 * 720 分辨率  （HDTV 720p）  
【opencv】：0.8985519409179688 ms  
【base64】：46.4552640914917 ms  
【PIL】：1.798558235168457 ms  
【Tensorflow】：4.8030853271484375 ms  
【mxnet】：23.89671802520752 ms  

1080 * 2396 分辨率   
【opencv】：45.20227909088135 ms  
【base64】：56.400203704833984 ms  
【PIL】：32.40470886230469 ms  
【Tensorflow】：100.76427459716797 ms  
【mxnet】：89.70215320587158 ms  

819 * 1024 分辨率  
【opencv】：23.30014705657959 ms  
【base64】：53.59981060028076 ms  
【PIL】：18.80052089691162 ms  
【Tensorflow】：47.302937507629395 ms  
【mxnet】：51.296281814575195 ms  

704 * 480 分辨率 （EDTV 480p）  
【opencv】：4.499578475952148 ms  
【base64】：48.1187105178833 ms  
【PIL】：4.200077056884766 ms  
【Tensorflow】：10.89944839477539 ms  
【mxnet】：40.57669639587402 ms  

640 * 480 分辨率   
【opencv】：6.051015853881836 ms  
【base64】：55.760979652404785 ms  
【PIL】：3.7998199462890625 ms  
【Tensorflow】：9.500336647033691 ms  
【mxnet】：30.202484130859375 ms  

250 * 250 分辨率  
【opencv】：1.1983633041381836 ms  
【base64】：47.60618209838867 ms  
【PIL】：2.003002166748047 ms  
【Tensorflow】：7.396864891052246 ms  
【mxnet】：25.00326633453369 ms  

- 对于常见分辨率，读取速度普遍较快（几毫秒），其中 opencv 的读取速度较快
- 对于非常见分辨率，读取速度普遍较慢（几十毫秒），其中 PIL 的读取速度较快


