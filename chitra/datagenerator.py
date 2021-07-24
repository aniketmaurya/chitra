import os
import pathlib
import random
import time
from functools import partial
from glob import glob
from pathlib import Path
from typing import Callable, Union

import tensorflow as tf
from typeguard import check_argument_types, typechecked

from .tf_image import read_image, resize_image


def benchmark(dataset, num_epochs=2, fake_infer_time=0.001):
    """Use this function to benchmark your Dataset loading time."""
    start_time = time.perf_counter()
    for _ in range(num_epochs):
        for _ in dataset:
            # Performing a training step
            time.sleep(fake_infer_time)
    tf.print(
        f"Execution time for {num_epochs} epochs: {time.perf_counter() - start_time :0.3f} seconds"
    )


def get_filenames(root_dir):
    root_dir = pathlib.Path(root_dir)
    return glob(str(root_dir / "*/*"))


def get_label(filename):
    return filename.split("/")[-2]


class ImageSizeList:
    def __init__(self, img_sz_list=None):

        if (
            isinstance(img_sz_list, (list, tuple))
            and len(img_sz_list) != 0
            and not isinstance(img_sz_list[0], (list, tuple))
        ):
            img_sz_list = [img_sz_list][:]

        self.start_size = None
        self.last_size = None
        self.curr_size = None
        self.img_sz_list = img_sz_list

        try:
            self.start_size = img_sz_list[0]
            self.last_size = img_sz_list[-1]
            self.curr_size = img_sz_list[0]
        except (IndexError, TypeError):
            print("No item present in the image size list")
            self.curr_size = None  # no item present in the list

    def get_size(self):
        img_sz_list = self.img_sz_list
        try:
            self.curr_size = img_sz_list.pop(0)
        except (IndexError, AttributeError):
            print(f"Returning the last set size which is: {self.curr_size}")

        return self.curr_size


# Cell
class Pipeline:
    @typechecked
    def __init__(self, funcs: Union[Callable, list, tuple] = None):
        if not check_argument_types():
            raise AssertionError
        if isinstance(funcs, list):
            self.funcs = funcs
        elif callable(funcs):
            self.funcs = [funcs]
        else:
            self.funcs = []

    @typechecked
    def add(self, func: Callable):
        if not check_argument_types():
            raise AssertionError
        self.funcs.append(func)

    def __call__(self, item):
        try:
            for func in self.funcs:
                item = func(item)
        except Exception as e:
            print("Error while applying function in pipeline!")
            raise e
        return item


class Dataset:
    MAPPINGS = {
        "PY_TO_TF": {str: tf.string, int: tf.int32, float: tf.float32},
    }

    def __init__(
        self,
        train_dir: Union[str, Path],
        image_size=None,
        transforms=None,
        default_encode=True,
        **kwargs,
    ):
        """
        Create a Dataset object that can generate tf.data.Dataset
        Args:
            train_dir:
            image_size:
            transforms:
            default_encode:
            **kwargs:
        """
        self.get_filenames = get_filenames
        self.read_image = read_image
        self.get_label = get_label
        self.transforms = transforms

        self.root_dir = train_dir
        self.default_encode = default_encode
        self.filenames = self.get_filenames(train_dir)
        self.num_files = len(self.filenames)
        self.image_size = image_size
        self.img_sz_list = ImageSizeList(self.image_size)

        self.labels = kwargs.get("labels", self.get_labels())

    def __len__(self):
        return len(self.filenames)

    def _process(self, filename):
        image = self.read_image(filename)
        label = self.get_label(filename)
        return image, label

    def _reload(self):
        image_size = self.image_size[:]
        self.filenames = self.get_filenames(self.root_dir)
        self.num_files = len(self.filenames)
        self.img_sz_list = ImageSizeList(image_size)
        self.labels = self.get_labels()

    def _capture_return_types(self):
        return_types = []
        for e in self.generator():
            outputs = e
            break
        if isinstance(outputs, tuple):
            for ret_type in outputs:
                return_types.append(
                    ret_type.dtype
                    if tf.is_tensor(ret_type)
                    else Dataset.MAPPINGS["PY_TO_TF"][type(ret_type)]
                )
        else:
            raise UserWarning("Unable to capture return type!")
        return tuple(return_types)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        return self._process(filename)

    def update_component(self, component_name, new_component):
        setattr(self, component_name, new_component)
        print(f"{component_name} updated with {new_component}")
        self._reload()

    def get_labels(self):
        # get labels should also update self.num_classes
        root_dir = self.root_dir
        labels = set()
        folders = glob(f"{root_dir}/*")
        for folder in folders:
            labels.add(os.path.basename(folder))

        labels = sorted(labels)
        self.NUM_CLASSES = len(labels)
        self.label_to_idx = {label: i for i, label in enumerate(labels)}

        return labels

    def label_encoder(self, label):
        idx = self.label_to_idx.get(label, None)
        if idx is None:
            raise AssertionError(f"Error while converting label={label} to index!")
        return idx

    def generator(self, shuffle=False):
        if shuffle:
            random.shuffle(self.filenames)
        img_sz = self.img_sz_list.get_size()
        n = len(self.filenames)
        for i in range(n):
            image, label = self.__getitem__(i)
            if img_sz:
                image = resize_image(image, img_sz)
            if self.transforms:
                image = self.transforms(image)
            if self.default_encode is True:
                label = self.label_encoder(label)
            yield image, label

    def get_tf_dataset(self, output_shape=None, shuffle=True):
        return_types = self._capture_return_types()
        self._reload()
        generator = partial(self.generator, shuffle=shuffle)
        datagen = tf.data.Dataset.from_generator(generator, return_types, output_shape)

        return datagen
