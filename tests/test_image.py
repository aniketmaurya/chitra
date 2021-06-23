from PIL import Image

from chitra.image import Chitra


def test__load_image():
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png'
    image = Chitra(url, cache=True)
    assert isinstance(image.image, Image.Image)


def test_numpy():
    assert True


def test_to_tensor():
    assert True


def test_shape():
    assert True


def test_size():
    assert True


def test_imshow():
    assert True


def test_draw_boxes():
    assert True
