from chitra.trainer import create_cnn


def test_create_cnn():
    cnn = create_cnn('mobilenetv2', 1, weights=None)
    assert cnn.trainable == True
