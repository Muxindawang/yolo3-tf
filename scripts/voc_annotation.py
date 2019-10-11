# 在./data/dataset 下加入了test_voc.txt

import os
import argparse
import xml.etree.ElementTree as ET


# 转换voc注释信息
def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path   # 注释信息
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')    # 找到所有object节点
            for obj in objects:
                difficult = obj.find('difficult').text.strip()   # 找到difficult节点
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    # difficult 是voc中的一个参数  如果被标记difficult则忽略它 这一般是非常小的物体
                    # voc中的另外一种标识 truncated
                    # voc数据集坐标从1开始
                    continue
                bbox = obj.find('bndbox')    # 找到bbox
                # 把obj中的name变为小写删除空格范围index 下标 用于确定类别
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/czl/PYPYPYPY/tensorflow-yolov3/voc")
    parser.add_argument("--train_annotation", default="../data/dataset/voc_train.txt")
    parser.add_argument("--test_annotation",  default="../data/dataset/voc_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation): os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation): os.remove(flags.test_annotation)
    # convert_voc_annotation 参数 data—path data-type anno_path
    num1 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2007'), 'trainval', flags.train_annotation, False)
    # flags.data_path='/home/czl/PYPYPYPY/tensorflow-yolov3/voc' 'train/VOCdevkit/VOC2007'
    num2 = convert_voc_annotation(os.path.join(flags.data_path, 'train/VOCdevkit/VOC2012'), 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotation(os.path.join(flags.data_path, 'test/VOCdevkit/VOC2007'),  'test', flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1 + num2, num3))


