import tensorflow as tf

def generate_edge_tensor(label, edge_width=3):
    if len(label.shape) == 2:
        label = tf.expand_dims(label,0)
    n, h, w = label.shape
    edge = tf.zeros(label.shape)
    # right
    edge_right1 = edge[:, 1:h, :]
    edge_right2 = tf.expand_dims(edge[ :,0, :], 1)
    edge_right1 = tf.where(tf.math.logical_not((label[:, 1:h, :] != label[ : ,:h - 1, :]) & (label[:, 1:h, :] != 255)
                & (label[:, :h - 1, :] != 255)), edge[:, 1:h, :], tf.ones(edge_right1.shape))
    edge = tf.concat([edge_right2, edge_right1], axis=1)

    # up
    edge_up1 = edge[:, :, :w - 1]
    edge_up2 = tf.expand_dims(edge[:, :, w-1], -1)
    edge_up1 = tf.where(tf.math.logical_not((label[:, :, :w - 1] != label[ :,:, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)), edge[:, :, :w - 1], tf.ones(edge_up1.shape))
    edge = tf.concat([edge_up1, edge_up2], axis=2)

    # upright
    edge_upright1 = edge[:, :h - 1, :w - 1]
    edge_upright2 = tf.expand_dims(edge[: ,h-1, w-1],-1)
    edge_upright2 = tf.expand_dims(edge_upright2,-1)
    edge_upright3 = tf.expand_dims(edge[:, h-1, :w-1],1)
    edge_upright4 = tf.expand_dims(edge[:, :h-1, w-1],-1)
    edge_upright1 = tf.where(tf.math.logical_not((label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)), edge[:, :h - 1, :w - 1], tf.ones(edge_upright1.shape))
    edge = tf.concat([edge_upright1, edge_upright4], axis=2)
    edge2 = tf.concat([edge_upright3, edge_upright2], axis=2)
    edge = tf.concat([edge, edge2], axis=1)

    # bottomright
    edge_bottomright1 = edge[:, :h - 1, 1:w]
    edge_bottomright2 = tf.expand_dims(edge[:, :h - 1, 0],-1)
    edge_bottomright3 = tf.expand_dims(edge[:, h - 1,:], 1)
    edge_bottomright1 = tf.where(tf.math.logical_not((label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)), edge[:, :h - 1, 1:w], tf.ones(edge_bottomright1.shape))
    edge = tf.concat([edge_bottomright2, edge_bottomright1], axis=2)
    edge = tf.concat([edge, edge_bottomright3], axis = 1)

    kernel = tf.ones((edge_width, edge_width, 1, 1), dtype=tf.float32)
    edge = tf.expand_dims(edge,-1)
    edge = tf.nn.conv2d(edge, kernel, strides=1, padding='SAME')
    edge = tf.where(tf.math.logical_not(edge!=0), edge, tf.ones(edge.shape))
    edge = tf.squeeze(edge,-1)
    return edge