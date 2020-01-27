# Chitra
> Library for creating data input pipeline in pure Tensorflow 2.x


.
<!-- This file will become your README and also the index of your documentation. -->

## Install

`pip install chitra`

## How to use

<!-- Fill me in please! Don't forget code examples: -->
## Loading data for image classification

```python
import tensorflow as tf
import chitra
from chitra.dataloader import Clf, show_batch

path = '/Users/aniketmaurya/Pictures/cats'

clf_dl = Clf()
data = clf_dl.from_folder(path)

print('class names:', clf_dl.CLASS_NAMES)

show_batch(data, 6, (6, 6))
```

    class names: (b'whitecat', b'blackcat')



![png](docs/images/output_5_1.png)


```python
model = tf.keras.applications.ResNet50(include_top=False,
                                              weights='imagenet',
                                              input_shape=(160, 160, 3),
                                              classes=2)
```

```python
# for e in data.batch(4): print(e)
```

```python
img = chitra.image.read_image('/Users/aniketmaurya/Pictures/cats/whitecat/wcat1.jpg')
img.shape
```




    TensorShape([683, 1024, 3])



```python
chitra.image.resize_image(img, (160, 160)).shape
```




    TensorShape([160, 160, 3])


