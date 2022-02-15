from itertools import cycle
import os
import json
import timeit
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import cv2

from networks.AugmentCE2P import ResNet
import utils.schp as schp
from datasets.target_generation import generate_edge_tensor
from utils.criterion import CriterionAll
from utils.callbacks import SGDRScheduler
from datasets.datasets import ATRDataset
from tensorflow.keras.layers import Normalization

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='val')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=str, default='512,512')
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--restore", type=str, default='Checkpoints/epoch_59/epoch_59.json')
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", action='store_false')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default=False)
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()



def main():
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0
    logs = {}
    if args.restore != None:
        try:
            # args.restore is path to json file checkpoint
            f = open(args.restore)
            logs = json.load(f)
            start_epoch = logs['epoch'] + 1
            losses = logs['l']
            if logs['epoch'] + 1 < 100:
                model = ResNet(num_classes=18)
                schp_model = ResNet(num_classes=18)
                model.build((None, 512, 512, 3))
                schp_model.build((None, 512, 512, 3))
                model.load_weights(logs['path_cp'])
                print('Load model checkpoint from: ', logs['path_cp'])
            else:
                cycle_n = int((logs['epoch']-99)/10+1)
                model = ResNet(num_classes=18)
                model.build((None, 512, 512, 3))
                schp_model = ResNet(num_classes=18)
                schp_model.build((None, 512, 512, 3))
                model.load_weights(logs['path_cp'])
                print('Load model checkpoint from: ', logs['path_cp'])
                schp_epoch = 100 + (cycle_n-1)*10 - 1
                schp_path = os.path.join('Checkpoints', 'schp_epoch_{}'.format(schp_epoch), 'schp_epoch_{}'.format(schp_epoch))
                schp_model.load_weights(schp_path)
                print('Load schp model checkpoint from: ', schp_path)
        except RuntimeError as e:
            print(e)
    else:
        model = ResNet(num_classes=18)
        schp_model = ResNet(num_classes=18)
        schp_model.build((None, 512, 512, 3))
        losses = []


    normalize = Normalization(mean=[0.406, 0.456, 0.485], variance=[0.225, 0.224, 0.229])

    def transforms(image, nor = normalize):
        """
        convert image to tensor, and normalize it with mean and std
        This is rewrite from the code:

        """
        img = tf.convert_to_tensor(image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        norm_img = nor(img)
        return norm_img

    
    # Model initialization

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)

    # Data Loader

    train_dataset = ATRDataset(root=args.data_dir, dataset='val', crop_size=[512,512], transform=transforms)
    train_data = train_dataset.load_data_train()
    train_data = train_data.shuffle(4096).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


    # Optimization
    opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum, nesterov=False, name="SGD")

    scheduler = SGDRScheduler()
    callbacks = tf.keras.callbacks.CallbackList(callbacks=[scheduler], model=model)

    model.compile(optimizer=opt, loss=criterion)

    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        return pred_mask

    def process_mask(mask):
        mask_np = mask.numpy().astype('float32')
        print('mask_np.shape:',mask_np.shape)
        mask_np = cv2.resize(mask_np, (512, 512))
        for i in range(18):
            mask_np[mask_np==i] = i*10
        mask_np = np.expand_dims(mask_np, -1)
        mask_3 = np.concatenate((mask_np, mask_np, mask_np), -1)
        return mask_3
    path_rs = 'results'
    # Training Loop
    for X in train_data:
        images, labels = X
        preds = model(images)
        preds_parsing = preds[0][0]
        preds_parsing = tf.compat.v1.image.resize_bilinear(images=preds_parsing, size=(512, 512), align_corners=True)
        print(preds_parsing.shape)
        for i in range(4):
            a = np.random.rand(1)
            a = float(a)
            a = str(a)
            mask = preds_parsing[i]
            mask_np = create_mask(mask)
            mask_3 = process_mask(mask_np)
            pred_mask = os.path.join(path_rs, a+'_pred.png')
            cv2.imwrite(pred_mask, mask_3)

            mask_label = labels[i]
            mask_3_label = process_mask(mask_label)
            label_mask = os.path.join(path_rs, a+'_label.png')
            cv2.imwrite(label_mask, mask_3_label)

if __name__=="__main__":
    main()