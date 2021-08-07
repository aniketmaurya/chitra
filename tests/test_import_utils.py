from chitra.import_utils import is_installed


def test_is_installed():
    assert is_installed("numpy")
