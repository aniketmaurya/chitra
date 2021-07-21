from chitra.utility.tf_utils import disable_gpu, gpu_dynamic_mem_growth, limit_gpu


def test_disable_gpu():
    assert disable_gpu() is None


def test_limit_gpu():
    assert limit_gpu(1, 1024) is None


def test_gpu_dynamic_mem_growth():
    assert gpu_dynamic_mem_growth() is None
