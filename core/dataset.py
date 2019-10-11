import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg


class Dataset(object):

    """
    继承object
    implement Dataset here
    """
    def __init__(self, dataset_type):
        # dataset_type 用于选择训练集和测试集
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        # TRAIN  self.input_sizes = 416
        # TEST   self.input_sizes = 544
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        # TRAIN self.batch_size = 4
        # TEST  self.batch_size = 2
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
        # TRAIN=True  参数可训练
        # TEST =False 参数冻结不可训练
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE # 416
        self.strides = np.array(cfg.YOLO.STRIDES)    # [8, 16, 32,]
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        # self.classes = 1 2 3 4 5 6 7 8 9 0 十个数字
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))   # [3,3,2]
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # anchors通过聚类求出，可以加快训练速度，但对最终训练精度无影响
        # anchors分为三个步长，每个步长包含三个尺寸的边框，实现不同精度的检验
        self.max_bbox_per_scale = 150   #每个尺寸下最多有150个候选框

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        # np.ceil向上取整，一个batch取多少样本 int（1000/4）=250
        # np.ceil 计算大于等于该值的最小值
        self.batch_count = 0

# 获取注释文本信息
    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            # len(line.strip().split()[1:]) != 0 表示非空白行
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        # iter next 连续使用 迭代数据 连续读取4（train batch size）张图片 然后return
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            # random.choice 从序列中随机选取一个元素
            self.train_output_sizes = self.train_input_size // self.strides
            # 输入图片的size
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            # 5 + self.num_classes 是 xywh+confidence anchor_per_scale = 3  这个是指每个锚点上3个anchor
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    # 循环所有batch中的每一张图片
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    # annotations[n] 的格式为 图片地址 + box1 位置（x1,y1,x2,y2）+类别....+boxn

                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image   # 将image样本导入batch-image 【4,416,416,3】
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox  #【4,52,52,3,85】
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        # 图片翻转 image shape = [416,416,3]
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            # 水平翻转 ::-1 步长-1
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
            # [:,[0,2]] 表示第二个维度上 0 和 2 索引对应的
            # bboxes shape 1*1*1*1*1*80  [[[[[[1,2...]]]]]]
            # w 图片宽度，如原xmin:6,max：8,w:10.翻转后  2,4
        return image, bboxes

    def random_crop(self, image, bboxes):
        # 随机裁剪
        if random.random() < 0.5:
            h, w, _ = image.shape
            # max_bbox 是用原bbox的min(xmin,ymin) max(xmax,ymax)画出的框 有两个边与原框重合
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # l left
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            # random.uniform 随机产生一个位于 x，y之间的实数
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            # 在边界框外面肆无忌惮的裁剪 保证边界框不会被剪切
            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
            # 最后再计算出新的边界框的 坐标
        return image, bboxes

    def random_translate(self, image, bboxes):
        # 对图片进行仿射变换
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            # 3opencv 中的仿射变换

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        # 注释的语法分析
        line = annotation.split()
        # /home/yang/test/TensorFlow2.0-Examples/4-Object_Detection/YOLOV3/data/dataset/test/000005.jpg （_处为一个空格）
        #_194,13,216,35,9 126,34,168,76,2 80,183,108,211,6 341,332,397,388,6
        image_path = line[0]
        # 图片地址
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
            # 若不存在路径 提示这个错误
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
        # 194,13,216,35,9   126,34,168,76,2   80,183,108,211,6   341,332,397,388,6

        if self.data_aug:
            # 训练的话 就执行下面的语句
            # image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    # ???????????????????????????????????
    #作用是将先验框转换到feature map同shape的label上
    def preprocess_true_boxes(self, bboxes):
        # self.train_output_size=[52,26,13]
        # self.anchor_per_scale=3
        # self.num_classes=80
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # 52*52*3*85  三个等级的anchor
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            # 坐标
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # 中心点坐标 w h 的长度 0.5*（xmax+xmin）   xmax-xmin
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




