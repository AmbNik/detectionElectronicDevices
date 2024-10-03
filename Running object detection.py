import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pyfirmata import Arduino, util
import time

namespace = {0: 'NOTHING', 1: 'sensor' , 2: 'motor'}
url = 'http://192.168.4.1'
port = "COM6"
board = Arduino(port)
servo_pin_y = 7  
servo_y = board.get_pin(f"d:{servo_pin_y}:s")
servo_pin_z = 8  
servo_z = board.get_pin(f"d:{servo_pin_z}:s")
servo_pin_x = 4 
servo_x = board.get_pin(f"d:{servo_pin_x}:s")
servo_pin_p = 4 
servo_p = board.get_pin(f"d:{servo_pin_p}:s")
angle_y = 110
servo_y.write(angle_y)
angle_z = 175
servo_z.write(angle_z)
angle_x = 70
servo_x.write(angle_x)
angle_p = 80
servo_p.write(angle_p)

localizator = tf.keras.models.load_model('my_bb_model.h5')
classifier = tf.keras.models.load_model('my_classifier.h5')

def Detect_objects(image):
    image = tf.cast(image, dtype = tf.float32) / 256
    small_image = tf.image.resize(image, (128, 128))
    big_image = tf.image.resize(image, (1024, 1024))
    image_exp = tf.expand_dims(small_image, axis = 0)
    bb_cords = localizator(image_exp)
    bb_cords = tf.squeeze(bb_cords, axis = 0)
    bb_cords = (bb_cords+1)/2*128
    bb_cords = tf.reshape(bb_cords, [10, 3])
    fxmin, ymin, fxmax = tf.split(bb_cords, 3, axis = 1)
    xmin = tf.minimum(fxmin, fxmax)
    xmax = tf.maximum(fxmin, fxmax)
    xmin = tf.clip_by_value(xmin, 0, 128)
    ymin = tf.clip_by_value(ymin, 0, 128)
    size = xmax - xmin
    xsize = tf.clip_by_value(size, 1, 128 - xmin)
    ysize = tf.clip_by_value(size, 1, 128 - ymin)
    ymin*= 8
    xmin*= 8
    ysize*= 8
    xsize*= 8
    for n in range(10):
        ii = tf.image.crop_to_bounding_box(big_image, int(ymin[n][0]), int(xmin[n][0]), int(ysize[n][0]), int(xsize[n][0]))
        ii = tf.image.resize(ii, (128,128))
        ii = tf.expand_dims(ii, axis = 0)
        if n == 0:
            cropped = ii 
        else:
            cropped = tf.concat([cropped, ii], axis = 0)
    probs = classifier(cropped)
    probs = probs.numpy()
    ma = np.amax(probs, axis = 1)
    ma = np.expand_dims(ma, axis = 1)
    _, classes = np.where(probs == ma)
    res_probs = []
    for a in range(10):
        res_probs.append(probs[a][classes[a]])
    cords = tf.concat([xmin/8, ymin/8, xmin/8+xsize/8, ymin/8+ysize/8], axis = 1)
    cords = cords.numpy()
    return cords, classes, res_probs
    
def visualize(in_image, cords, classes, probs, th = 0.8):
    big_image = tf.image.resize(in_image, (1024,1024)).numpy() / 256
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    for i in range(len(cords)):
        if classes[i]!= 0 and probs[i] >= th:
            if classes[i] == 1:
                color = (1, 0, 0)
            if classes[i] == 2:
                color = (0, 1, 0)
            text = namespace[classes[i]] + ' ' + str(probs[i]*100) + '%'    
            org = ((int(cords[i][0])*8,  int(cords[i][1])*8-10))
            big_image = cv2.putText(big_image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
            big_image = cv2.rectangle(big_image ,(int(cords[i][0])*8, int(cords[i][1])*8),(int(cords[i][2])*8,int(cords[i][3])*8),color,5)
    return big_image
   
def prettify(cords, classes, probs, tau = 0.5):
    newcords = []
    newclasses = []
    newprobs = []
    for i1 in range(len(classes)):
        if classes[i1]!=0:
            found = False
            for i2 in range(len(classes)):
                if classes[i2]!=0 and  i1 != i2:
                    x_overlap = max(0, min(cords[i1][2], cords[i2][2]) - max(cords[i1][0], cords[i2][0]))
                    y_overlap = max(0, min(cords[i1][3], cords[i2][3]) - max(cords[i1][1], cords[i2][1]))
                    inter = x_overlap*y_overlap
                    area1 = (cords[i1][2] - cords[i1][0])*(cords[i1][3] - cords[i1][1])
                    area2 = (cords[i2][2] - cords[i2][0])*(cords[i2][3] - cords[i2][1])
                    union = area1+area2 - inter
                    IoU = inter/union
                    if IoU > tau:
                        newcord = [(cords[i1][0]+ cords[i2][0])//2,(cords[i1][1]+ cords[i2][1])//2,(cords[i1][2]+ cords[i2][2])//2,(cords[i1][3]+ cords[i2][3])//2]
                        newcords.append(newcord)
                        newclasses.append(classes[i1])
                        newprobs.append(probs[i1])
                        classes[i1] = 0
                        classes[i2] = 0
                        found = True
            if found == False:
                newcords.append(cords[i1])
                newclasses.append(classes[i1])
                newprobs.append(probs[i1])
    return newcords, newclasses, newprobs
	
def perform_action(class_name):
	time.sleep(0.5)
    if class_name == 1:
        time.sleep(0.5)
        servo_y.write(65)
        time.sleep(0.5)
        servo_z.write(180)
        time.sleep(0.5)
        servo_x.write(90)
        time.sleep(0.5)
        servo_y.write(110)
        time.sleep(0.5)
        servo_x.write(70)
        time.sleep(0.5)
        servo_z.write(175)
        time.sleep(0.5)
    elif class_name == 2:
        time.sleep(0.5)
        servo_y.write(65)
        time.sleep(0.5)
        servo_z.write(180)
        time.sleep(0.5)
        servo_x.write(50)
        time.sleep(0.5)
        servo_y.write(110)
        time.sleep(0.5)
        servo_x.write(70)
        time.sleep(0.5)
        servo_z.write(175)
        time.sleep(0.5)
	
while True:
    stream = urllib.request.urlopen(url)
    bytes = b''
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cords, classes, probs = Detect_objects(frame)
            for i in range(3):
                cords, classes, probs = prettify(cords, classes, probs, 0)  
            result = visualize(frame, cords, classes, probs, 0.9)  
            cv2.imshow('frame', result)
			perform_action(classes[0])
            if cv2.waitKey(1) == ord('q'):
                break
cv.destroyAllWindows()