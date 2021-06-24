# Play with Images

> `Chitra` is an image utility class that can load image from filelike object, web url or numpy image. It offers drawing bounding box over the image.


```python
# pip install -U chitra==0.1.0a1

from chitra.image import Chitra
import matplotlib.pyplot as plt
```

## What can it do?
- Load image from file, [filelike object](https://docs.python.org/3/glossary.html#term-file-like-object), web url, or numpy array
- Plot image
- Plot bounding boxes along with labels in no extra code.
- Specify bounding box format:
       - **Center(xywh):** center x,y and height width of bbox
       - **Corner(xyxy):** xmin ymin and xmax ymax
- Plot bounding box on image


### Load image from web url and show

```python
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png'
image = Chitra(url)
image.imshow()
```

![png](output_6_0.png)

You can cache the image downloaded from internet URL by passing `cache=True` in argument.
Second call to the same URL will not download from internet, instead image will be loaded from the local cache dir.
```python
# first call
image = Chitra(url, cache=True)

# second call
image = Chitra(url, cache=True)
```


### Plot bounding box and label for the handphone

```python
box = [[600, 250, 900, 600.1]]
label = ['hand']
image = Chitra(url, box, label)
image.image = image.image.convert('RGB')
plt.imshow(image.draw_boxes())
```

![png](output_8_1.png)
