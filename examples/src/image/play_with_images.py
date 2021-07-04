import matplotlib.pyplot as plt
import rich

from chitra import Chitra

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png"
box = [[600, 250, 900, 600.1]]
label = ["handphone"]

image = Chitra(url, box, label)
rich.print("before resize:", image.bboxes)

image.resize_image_with_bbox((224, 224))
rich.print("after resize:", image.bboxes)

plt.imshow(image.draw_boxes())
