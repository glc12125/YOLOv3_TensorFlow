# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
import glob
import os

from model import yolov3

#DATA_PATH = '../cam_radar_sensor_fusion/tests/output/'

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image_path", type=str,
                    help="The path of the input image.")
parser.add_argument('--scene_sequences', type=str, default="0061,", help='The scene seuqneces', required=True)
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

images_path = args.input_image_path

scene_sequences = options.scene_sequences
scene_sequences = scene_sequences.split(',')
for sequence in scene_sequences:
    if len(sequence) == 4:
        scene_sequence = sequence
        data_path = images_path + scene_sequence + '/2d_detection'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        img_files = [f for f in glob.glob(images_path + "/{}/images/cam_*.jpg".format(scene_sequence), recursive=False)]        

        with tf.Session() as sess:
            yolo_model = yolov3(args.num_class, args.anchors)
            input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(input_data, False)
            saver = tf.train.Saver()
            saver.restore(sess, args.restore_path)  
            for img_path in img_files:
                img_ori = cv2.imread(img_path)
                height_ori, width_ori = img_ori.shape[:2]
                if args.letterbox_resize:
                    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
                else:
                    img = cv2.resize(img_ori, tuple(args.new_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.asarray(img, np.float32)
                img = img[np.newaxis, :] / 255.         

                pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)          

                pred_scores = pred_confs * pred_probs           

                boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)          

                boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})   
                detection_labels = []
                # rescale the coordinates to the original image
                if args.letterbox_resize:
                    boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                    boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
                else:
                    boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
                    boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))           

                print("box coords:")
                print(boxes_)
                print('*' * 30)
                print("scores:")
                print(scores_)
                print('*' * 30)
                print("labels:")
                print(labels_)          

                for i in range(len(boxes_)):
                    x0, y0, x1, y1 = boxes_[i]
                    top = max(0, np.floor(y0 + 0.5).astype('int32'))
                    left = max(0, np.floor(x0 + 0.5).astype('int32'))
                    bottom = min(height_ori, np.floor(y1 + 0.5).astype('int32'))
                    right = min(width_ori, np.floor(x1 + 0.5).astype('int32'))
                    print(args.classes[labels_[i]], (left, top), (right, bottom))    
                    detection_labels.append([args.classes[labels_[i]], scores_[i], left, top, right, bottom])
                    plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
                index = int(img_path.split('/')[-1][4:-4])
                with open(data_path + '/result_{}.txt'.format(index), 'w') as f:
                    for item in detection_labels:
                        f.write("%s\n" % item)
                cv2.imshow('Detection result', img_ori)
                cv2.imwrite('detection_result.jpg', img_ori)
                cv2.waitKey(1)
