from tensorflow.keras.losses import Loss
import tensorflow as tf
from datasets.target_generation import generate_edge_tensor

class ConsistencyLoss(Loss):
    """
    label: shape (b,512,512)
    pred: shape [parsing, edge] where parsing is [b,512,512,18], edge is [b,512,512,2]
    """
    def __init__(self, ignore_index = 255):
        super(ConsistencyLoss, self).__init__()
        self.ignore_index=ignore_index

    def call(self, label, pred):
        parsing = pred[0]
        edge = pred[1]
        parsing_pre = tf.argmax(parsing, axis=3) # convert to same format with label [b,512,512]
        parsing_pre = tf.where(label!=255, parsing_pre, tf.ones_like(parsing_pre)*255) 
        generated_edge = generate_edge_tensor(parsing_pre) # [b,512,512]
        edge_pre = tf.argmax(edge, axis=3) # convert to format [b, 512, 512]
        generated_edge = tf.cast(generated_edge, dtype=tf.int64)
        edge_pre = tf.cast(edge_pre, dtype=tf.int64)
        v_generated_edge = tf.where(label==255, tf.zeros_like(edge_pre), generated_edge)
        v_edge_pre = tf.where(label==255, tf.zeros_like(edge_pre), edge_pre)
        union_mask = (v_generated_edge==1)|(v_edge_pre==1)
        mask_generated_edge = tf.where(union_mask, v_generated_edge, tf.zeros_like(v_generated_edge))
        mask_edge_pre = tf.where(union_mask, v_edge_pre, tf.zeros_like(v_edge_pre))
        l = tf.keras.losses.Huber()
        return l(mask_generated_edge, mask_edge_pre)