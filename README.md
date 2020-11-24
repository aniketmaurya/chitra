# chitra
> <p align="center">


## What is chitra?

**chitra** (**चित्र**) is an image utility library for Deep Learning tasks. *(It is not image-processing library)*

chitra reduces image data loading boilerplates for classification and object-detection.

It can also generate bounding-boxes from the annotated dataset.

If you have more use cases please [**raise an issue**](https://github.com/aniketmaurya/chitra/issues/new/choose) with the feature you want.

## Installation

### Using pip (recommended)

`pip install -U chitra`

### From source

```
git clone https://github.com/aniketmaurya/chitra.git
cd chitra
pip install -e .
```

## Usage

### Loading data for image classification



```
import numpy as np
import tensorflow as tf
import chitra
from chitra.dataloader import Clf, show_batch
import matplotlib.pyplot as plt
```

```
clf_dl = Clf()
data = clf_dl.from_folder(cat_dog_path, target_shape=(224, 224))

clf_dl.show_batch(8, figsize=(8,8))
```

    CLASSES ENCODED: {'cat': 0, 'dog': 1}



![png](docs/images/output_4_1.png)


```
for e in data.take(1):
    image = e[0].numpy().astype('uint8')
    label = e[1].numpy()
plt.imshow(image)
plt.show()
```


![png](docs/images/output_5_0.png)


## Visualization

### Image annotation

Thanks to [**fizyr**](https://github.com/fizyr/keras-retinanet) keras-retinanet.

```
from chitra.visualization import draw_annotations

labels = np.array([label])
bbox = np.array([[30, 50, 170, 190]])
label_to_name = lambda x: 'Cat' if x==0 else 'Dog'
```

```
draw_annotations(image, ({'bboxes': bbox, 'labels':labels,}), label_to_name=label_to_name)
plt.imshow(image)
plt.show()
```


![png](docs/images/output_8_0.png)


## Image datagenerator
Dataset class provides the flexibility to load image dataset by updating components of the class.

Components of Dataset class are:
- image file generator
- resizer
- label generator
- image loader

These components can be updated with custom function by the user according to their dataset structure. For example the Tiny Imagenet dataset is organized as-

```
train_folder/
.....folder1/
    .....file.txt
    .....folder2/
           .....image1.jpg
           .....image2.jpg
                     .
                     .
                     .
           ......imageN.jpg
                    
                      
```

The inbuilt file generator search for images on the `folder1`, now we can just update the `image file generator` and rest of the functionality will remain same.

**Dataset also support progressive resizing of images.**

```
from chitra.datagenerator import Dataset
from glob import glob
```

### Updating component

```
# data_path = '/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train'
ds = Dataset(data_path)
# it will load the folders and NOT images
ds.filenames[:3]
```

    No item present in the image size list





    ['/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train/n03584254',
     '/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train/n02403003',
     '/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train/n02056570']



```
def load_files(path):
    return glob(f'{path}/*/images/*')

def get_label(path):
    return path.split('/')[-3]
    
ds.update_component('get_filenames', load_files)
ds.filenames[:3]
```

    get_filenames updated with <function load_files at 0x7fe0a5aa9560>
    No item present in the image size list





    ['/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train/n03584254/images/n03584254_251.JPEG',
     '/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train/n03584254/images/n03584254_348.JPEG',
     '/data/aniket/tiny-imagenet/data/tiny-imagenet-200/train/n03584254/images/n03584254_465.JPEG']



### Progressive resizing

```
image_sz_list = [(28, 28), (32, 32), (64, 64)]

ds = Dataset(data_path, image_size=image_sz_list)
ds.update_component('get_filenames', load_files)
ds.update_component('get_label', get_label)


print()
# first call to generator
for img, label in ds.generator():
    print('first call to generator:', img.shape)
    break

# seconds call to generator
for img, label in ds.generator():
    print('seconds call to generator:', img.shape)
    break

# third call to generator
for img, label in ds.generator():
    print('third call to generator:', img.shape)
    break

```

    get_filenames updated with <function load_files at 0x7fe0a5aa9560>
    get_label updated with <function get_label at 0x7fe14838ab00>
    
    first call to generator: (28, 28, 3)
    seconds call to generator: (32, 32, 3)
    third call to generator: (64, 64, 3)


### tf.data support
Creating a `tf.data` dataloader was never as easy as this one liner. It converts the Python generator into `tf.data.Dataset` for a faster data loading, prefetching, caching and everything provided by tf.data.

```
image_sz_list = [(28, 28), (32, 32), (64, 64)]

ds = Dataset(data_path, image_size=image_sz_list)
ds.update_component('get_filenames', load_files)
ds.update_component('get_label', get_label)

dl = ds.get_tf_dataset()

for e in dl.take(1):
    print(e[0].shape)

for e in dl.take(1):
    print(e[0].shape)

for e in dl.take(1):
    print(e[0].shape)
```

    get_filenames updated with <function load_files at 0x7fe0a5aa9560>
    get_label updated with <function get_label at 0x7fe14838ab00>
    (28, 28, 3)
    (32, 32, 3)
    (64, 64, 3)


## Learner
The Learner class inherits from `tf.keras.Model`, it contains everything that is required for training.

```
from chitra.learner import Learner
from chitra.datagenerator import Dataset
from PIL import Image
```

```
ds = Dataset(cat_dog_path, image_size=(224,224))
```

```
learner = Learner(ds, tf.keras.applications.MobileNetV2)
```

    WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.


```
learner.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    mobilenetv2_1.00_224 (Functi (None, None, None, 1280)  2257984   
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 1280)              0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1280)              0         
    _________________________________________________________________
    output (Dense)               (None, 1)                 1281      
    =================================================================
    Total params: 2,259,265
    Trainable params: 2,225,153
    Non-trainable params: 34,112
    _________________________________________________________________


```
learner.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])
```

```
learner.cyclic_fit(epochs=10,
                   batch_size=1,
                   lr_range=(0.01, 0.01)
                  )
```

    Returning the last set size which is: (224, 224)
    Epoch 1/10
    8/8 [==============================] - 1s 109ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 2/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 109ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 3/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 112ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 4/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 111ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 5/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 108ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 6/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 108ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 7/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 109ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 8/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 115ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 9/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 116ms/step - loss: 7.7125 - binary_accuracy: 0.5000
    Epoch 10/10
    Returning the last set size which is: (224, 224)
    8/8 [==============================] - 1s 117ms/step - loss: 7.7125 - binary_accuracy: 0.5000





    <tensorflow.python.keras.callbacks.History at 0x7fb21487d2d0>



## Model Visualization

```
from chitra.learner import InterpretModel
```

```
learner = Learner(ds, tf.keras.applications.Xception, include_top=True)
model_interpret = InterpretModel(False, learner)
```

```
image = ds[1][0].numpy().astype('uint8')
image = Image.fromarray(image)
model_interpret(image)
```

    Returning the last set size which is: (224, 224)
    index: 285



![png](docs/images/output_28_1.png)


```
IMAGENET_LABELS[285]
```




    'Egyptian Mau'



```
image = ds[3][0].numpy().astype('uint8')
image = Image.fromarray(image)
print(IMAGENET_LABELS[208])
model_interpret(image)
```

    Labrador Retriever
    Returning the last set size which is: (224, 224)
    index: 208



![png](docs/images/output_30_1.png)


## Utils

Limit GPU memory or enable dynamic GPU memory growth for Tensorflow

```
from chitra.utils import limit_gpu, gpu_dynamic_mem_growth

# limit the amount of GPU required for your training
limit_gpu(gpu_id=0, memory_limit=1024*2)
```

    No GPU:0 found in your system!


```
gpu_dynamic_mem_growth()
```

    No GPU found on the machine!


## Contributing

Contributions of any kind are welcome. Please check the [**Contributing Guidelines**](https://github.com/aniketmaurya/chitra/blob/master/CONTRIBUTING.md) before contributing.
