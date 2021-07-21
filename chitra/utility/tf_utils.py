import os

import tensorflow as tf


def disable_gpu():
    """disable gpu for tensorflow."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)


def limit_gpu(gpu_id: int, memory_limit: int):
    """limit the selected gpu [gpu_id] by [memory_limit] MB."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) >= gpu_id + 1:
        raise AssertionError
    if gpus:
        # Restrict TensorFlow to only allocate [memory MB] of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_id],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )
                ],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print(f"No GPU:{gpu_id} found in your system!")


def gpu_dynamic_mem_growth():
    """Borrowed from
    https://github.com/philipperemy/keract/tree/master/examples.

    Check for GPUs and set them to dynamically grow memory as needed
    Avoids OOM from tensorflow greedily allocating GPU memory
    """
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        if len(gpu_devices) > 0:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU dynamic memory growth enabled")
        else:
            print("No GPU found on the machine!")
    except AttributeError:
        print(
            "Upgrade your tensorflow to 2.x to have the gpu_dynamic_mem_growth feature."
        )
