import tensorflow as tf

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.trainable_variables, net2.trainable_variables):
        param1.assign(param1*((1.0-alpha)+alpha*param2))

def set_momentum(model, momentum):
    # In PSP_module
    model.context_encoding.bn.momentum = momentum
    # In Edge_module
    model.edge.bn1.momentum = momentum
    model.edge.bn2.momentum = momentum
    model.edge.bn3.momentum = momentum
    # In Decoder_module
    model.decoder.bn1.momentum = momentum
    model.decoder.bn2.momentum = momentum
    model.decoder.bn3.momentum = momentum
    model.decoder.bn4.momentum = momentum
    # In Resent module
    model.fusion.layers[1].momentum = momentum
    
def reset_bn(model):
    # Set all momentum ABN layer to 0
    set_momentum(model, 0)

def reset_moving_mean_and_var(model):
    """ 
    This function aims to reset moving_mean and moving_var to default 0,1
    by set momentum = 0 for all ABN layers in model and feeding to model a fake batch data which has mean=0, and std=1
    Formulas:
    moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)
    moving_var = moving_var * momentum + var(batch) * (1 - momentum)

    And then set momentum to 1, and start to learn moving_mean and moving_var from train dataset
    """
    reset_bn(model)
    fake_data = tf.random.normal((8,512,512,3))
    model(fake_data)
    set_momentum(model, 1)



def bn_re_estimate(loader, model):
    n = 0
    reset_moving_mean_and_var(model)
    for data in loader:
        images, labels = data
        b = images.shape[0]
        momentum = b / (n + b)
        set_momentum(model, momentum=momentum)
        model(images)
        n += b
    set_momentum(model, momentum=momentum)



    

    

