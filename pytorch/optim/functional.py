def optimizer_scheduler(optimizer, p,closuer=None):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    lr = 0.01 / (1. + 10 * p) ** 0.75

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

