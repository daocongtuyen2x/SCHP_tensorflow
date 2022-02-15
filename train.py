from itertools import cycle
import os
import json
import timeit
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np

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
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='ATR')
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()



def main():
    args = get_arguments()
    print(args)

    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    try:
        # Currently, memory growth needs to be the same across GPUs
        tf.config.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Using ", gpus[1], ", ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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
                schp_path = os.path.join('checkpoints', 'schp_epoch_{}'.format(schp_epoch), 'schp_epoch_{}'.format(schp_epoch))
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

    train_dataset = ATRDataset(root=args.data_dir, dataset='train', crop_size=[512,512], transform=transforms)
    train_data = train_dataset.load_data_train()
    train_data = train_data.shuffle(4096).batch(8, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


    # Optimization
    opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum, nesterov=False, name="SGD")
    scheduler = SGDRScheduler()
    callbacks = tf.keras.callbacks.CallbackList(callbacks=[scheduler], model=model)
    model.compile(optimizer=opt, loss=criterion)

    # Training Loop

    callbacks.on_train_begin(start_epoch)
    for epoch in range(start_epoch, args.epochs):
        losses_list = []
        with tqdm(train_data) as pbar:
            pbar.set_description(f"[Epoch {epoch}]")
            for step, X in enumerate(pbar):

                images, labels = X
                edges = generate_edge_tensor(labels)
                # Online Self Correction Cycle with Label Refinement
                if cycle_n >= 1:
                    soft_preds = [schp_model(images)]
                    soft_parsing = []
                    soft_edge = []
                    for soft_pred in soft_preds:
                        soft_parsing.append(soft_pred[0][-1])
                        soft_edge.append(soft_pred[1][-1])
                    soft_preds = tf.concat(soft_parsing, axis=0)
                    soft_edges = tf.concat(soft_edge, axis=0)
                else:
                    soft_preds = None
                    soft_edges = None
                
                with tf.GradientTape() as d_tape:
                    preds = model(images)
                    loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)
                # Calculate gradient
                gradients = d_tape.gradient(loss, model.trainable_variables)
                # Update params
                opt.apply_gradients(zip(gradients, model.trainable_variables))
                losses_list.append(loss.numpy())

            # Self Correction Cycle with Model Aggregation
            if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
                print('Self-correction cycle number {}'.format(cycle_n))
                schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
                cycle_n += 1
                schp.bn_re_estimate(train_data, schp_model)
                folder_schp = os.path.join('checkpoints', 'schp_epoch_{}'.format(epoch))
                schp_model.save_weights(os.path.join(folder_schp,'schp_epoch_{}'.format(epoch)))
        losses.append(float(np.mean(losses_list)))
        callbacks.on_epoch_end(epoch, losses)
    callbacks.on_train_end()
if __name__=="__main__":
    main()