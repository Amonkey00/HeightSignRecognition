import time
import os
from absl import app, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from .yolov3_tf2.models import (
    YoloV3
)
from .yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from .patchNet import models as model
from .OCR import ocr_getResult
from django.conf import settings


sign_classes = "static/data/roundSign.names"
size = 416
num_classes = 1
weights = "static/checkpoints/yolov3_Sign5.tf"
patchNet_weights = "static/checkpoints/PatchNet_v2.tf"
output_path = "images\\"

# Avoid Out of GPU memory
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Get yolov3 model
yolo=YoloV3(classes=num_classes)

# Load Weights of yolov3_Sign
yolo.load_weights(weights).expect_partial()
print('---------------Yolov3 weights loaded---------------')

# Get class names
class_names = [c.strip() for c in open(sign_classes).readlines()]
print('---------------classes loaded---------------')

# Get PatchNet model
patchNet = model.PatchNet(num_classes=2)

# Load Weights of PatchNet
patchNet.load_weights(patchNet_weights).expect_partial()
print('---------------PatchNet weights loaded---------------')


def CNN_recognize(img_path,img_name,img_save=True):

    # convert .jpg to input data
    img_raw = tf.image.decode_image(
        open(img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img, signs = draw_signs(img, (boxes, scores, classes, nums), class_names, patchNet)
    if img_save:
        print(os.path.join(settings.STATIC_ROOT,output_path+img_name))
        cv2.imwrite(os.path.join(settings.STATIC_ROOT,output_path+img_name),img)


def getNumberFromResult(ocr_result):
    nums = []
    for recog_part in ocr_result['ret']:
        word = recog_part['word']
        for c in word:
            if c.isdigit():
                nums.append(float(c))
                if len(nums)==2:
                    return str(nums[0]+0.1*nums[1])
    if len(nums)==1:
        return str(nums[0])
    return "Hard to recognize"


def draw_signs(img, outputs, class_names, PatchNet):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    signs = []
    sign_images = []
    predict_images = []
    for i in range(nums):
        x1y1 = (np.array(boxes[i][0:2]) * wh).astype(np.int32)
        x2y2 = (np.array(boxes[i][2:4]) * wh).astype(np.int32)
        sign_img = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        sign_img = cv2.resize(sign_img,(112,112))
        sign_images.append(sign_img)
        predict_img = sign_img / 255
        predict_images.append(predict_img)

    if len(sign_images) > 0:
        predict_images = np.array(predict_images)
        judges = np.argmax(PatchNet.predict(predict_images), axis=1)
        for idx, judge in enumerate(judges):
            if judge == 1:
                signs.append(sign_images[idx])
                tmpSign_path = os.path.join(settings.STATIC_ROOT, 'tmpSign.jpg')
                cv2.imwrite(tmpSign_path, sign_images[idx])
                SignText = getNumberFromResult(ocr_getResult(tmpSign_path))
                os.remove(tmpSign_path)
                x1y1 = (np.array(boxes[idx][0:2]) * wh).astype(np.int32)
                x2y2 = (np.array(boxes[idx][2:4]) * wh).astype(np.int32)
                img = cv2.rectangle(img, tuple(x1y1), tuple(x2y2), (255, 0, 0), 2)
                img = cv2.putText(img, '{} {}'.format(
                    'Sign', SignText+'M'),
                                  tuple(x1y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    return img, signs


CNN_recognize("static/data/P4.jpg",img_name="preload",img_save=False)
