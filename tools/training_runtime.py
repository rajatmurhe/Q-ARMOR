"""Select TensorFlow execution resources before building any model."""


def configure_runtime(device='auto', threads=4):
    import tensorflow as tf
    if device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    elif device == 'metal' and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('Metal requested but TensorFlow has no GPU device')
    if threads < 1:
        raise ValueError('threads must be positive')
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    return {'device_requested': device, 'threads': threads,
            'visible_devices': [str(d) for d in tf.config.get_visible_devices()]}
