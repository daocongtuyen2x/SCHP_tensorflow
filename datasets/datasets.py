import os
import numpy as np
import random
import cv2
import tensorflow as tf
from utils.transforms import get_affine_transform
class ATRDataset():
    def __init__(self, root , dataset, crop_size=[512,512], scale_factor = 0.25,
                 rotation_factor = 30, ignore_label = 255, transform=None):
        super(ATRDataset, self).__init__()
        self.root = root
        self.aspect_ratio = crop_size[1] *1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.transform = transform
        self.dataset = dataset

        """
        root --  "G:\\retain\AIP391\VIRTUAL_TRY_ON\dataset\ATR"
        dataset -- train/val/test
        """
        list_path = os.path.join(self.root, self.dataset + '_id.txt')
        train_list = [i_id.strip() for i_id in open(list_path)]

        self.train_list = train_list
        self.number_samples = len(self.train_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale


        #map processing to data

    def processing(self ,tpl):
        # print(tpl[0], tpl[1])
        # return tf.convert_to_tensor(tpl[0]), tf.convert_to_tensor(tpl[1])
        im_path = tpl[0].numpy().decode("utf-8")
        # print(im_path)
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            label_path = tpl[1].numpy().decode("utf-8")
            parsing_anno = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if self.dataset == 'train' or self.dataset == 'val':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    person_center[0] = im.shape[1] - person_center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        if self.dataset == 'test':
            return tf.convert_to_tensor(im), tf.convert_to_tensor(parsing_anno)
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))
            label_parsing = tf.convert_to_tensor(label_parsing)

            # print('success')
            return input, label_parsing

    def load_data_train(self):
        # train_item = self.train_list[index]
        data = []
        for i in self.train_list:
            data.append((os.path.join(self.root, self.dataset + '_images', i + '.jpg'),
                        os.path.join(self.root, self.dataset + '_segmentations', i + '.png')))

        train_image_ds = tf.data.Dataset.from_tensor_slices(data)

        train_ds = train_image_ds.map(lambda tpl: tf.py_function(self.processing, [tpl], [tf.float32, tf.uint8]))

        return train_ds
        
if __name__=='__main__':
    data_dir = "ATR"
    train_dataset = ATRDataset(root=data_dir, dataset='train',crop_size=[512,512], transform=transforms)
    train = train_dataset.load_data_train()

    train = train.shuffle(4096).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    for i in train:
        print(i[0], i[1])
        break