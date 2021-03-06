import os

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

flags.DEFINE_string('data_dir', 'D:/SignData/',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val spit')
flags.DEFINE_string('output_train', 'D:/SignData/sign_train.tfrecord', 'output train dataset')
flags.DEFINE_string('output_val', 'D:/SignData/sign_val.tfrecord', 'output validation dataset')
flags.DEFINE_string('classes', '../data/roundSign.names', 'classes file')
flags.DEFINE_enum('index','0', [
                    '0','1','2'], 'index marks the dataset labeled by who')


# Build example of an image
def build_example(annotation, class_map):
    img_path = os.path.join(
        FLAGS.data_dir, 'JPEGImages', annotation['filename'])
    img_raw = open(img_path, 'rb').read()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    if 'object' in annotation:
        for obj in annotation['object']:

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
    }))
    return example


# Parse the .xml File
def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_argv):
    class_map = {name: idx for idx, name in enumerate(
        open(FLAGS.classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)

    writer = tf.io.TFRecordWriter(FLAGS.output_train)
    image_list = open(os.path.join(
        FLAGS.data_dir, 'ImageSets', 'Main', 'sign_%s.txt' % 'train')).read().splitlines()
    logging.info("Train image list loaded: %d", len(image_list))
    for image in tqdm.tqdm(image_list):
        name = image
        annotation_xml = os.path.join(
            FLAGS.data_dir, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']

        # Ignore image that doesn't contain roungSign?
        if 'object' not in annotation:
            continue
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()

    writer = tf.io.TFRecordWriter(FLAGS.output_val)
    image_list = open(os.path.join(
        FLAGS.data_dir, 'ImageSets', 'Main', 'sign_%s.txt' % 'val')).read().splitlines()
    logging.info("Validation image list loaded: %d", len(image_list))
    for image in tqdm.tqdm(image_list):
        name = image
        annotation_xml = os.path.join(
            FLAGS.data_dir, 'Annotations', name + '.xml')
        annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
        annotation = parse_xml(annotation_xml)['annotation']

        # Ignore image that doesn't contain roungSign?
        if 'object' not in annotation:
            continue
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()

    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
