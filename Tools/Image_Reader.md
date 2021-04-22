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
【opencv】：36.552181243896484 ms  
【base64】：39.679646492004395 ms  
【PIL】：22.949984073638916 ms  
【Tensorflow】：104.11935329437256 ms  
【mxnet】：61.89755201339722 ms

1366 * 768 分辨率 （HDMI）  
【opencv】：19.682440757751465 ms  
【base64】：28.839659690856934 ms  
【PIL】：12.310023307800293 ms  
【Tensorflow】：97.51430988311768 ms  
【mxnet】：35.186288356781006 ms

1280 * 720 分辨率  （HDTV 720p）  
【opencv】：21.622326374053955 ms  
【base64】：22.880887985229492 ms  
【PIL】：12.010011672973633 ms  
【Tensorflow】：87.76040315628052 ms  
【mxnet】：32.549946308135986 ms  

1080 * 2396 分辨率   
【opencv】：41.41058921813965 ms  
【base64】：44.73029613494873 ms  
【PIL】：27.700233459472656 ms  
【Tensorflow】：111.29999160766602 ms  
【mxnet】：62.840611934661865 ms  

704 * 480 分辨率 （EDTV 480p）  
【opencv】：8.377723693847656 ms  
【base64】：6.480274200439453 ms  
【PIL】：6.359999179840088 ms  
【Tensorflow】：19.020025730133057 ms  
【mxnet】：13.225083351135254 ms  

640 * 480 分辨率   
【opencv】：10.829606056213379 ms  
【base64】：10.0700044631958 ms   
【PIL】：7.929985523223876 ms  
【Tensorflow】：59.81001377105713 ms  
【mxnet】：18.16997528076172 ms  

250 * 250 分辨率  
【opencv】：1.3496184349060059 ms  
【base64】：1.7699122428894043 ms  
【PIL】：1.0100173950195312 ms  
【Tensorflow】：12.31121301651001 ms  
【mxnet】：4.74010705947876 ms  




