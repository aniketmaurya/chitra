from chitra.import_utils import is_installed

onnx = None
tf2onnx = None
onnx2pytorch = None
torch = None

if is_installed("onnx"):
    import onnx

if is_installed("tf2onnx"):
    import tf2onnx

if is_installed("torch"):
    import torch.onnx


def pytorch_to_onnx(model, tensor, export_path="temp.onnx"):
    # Input to the model
    torch_out = model(tensor)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        tensor,  # model input (or a tuple for multiple inputs)
        export_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    return export_path


def tf2_to_onnx(model, opset=None, output_path=None, **kwargs):
    inputs_as_nchw = kwargs.get("inputs_as_nchw", "input0:0")
    onnx_model = tf2onnx.convert.from_keras(
        model, opset=opset, output_path=output_path, inputs_as_nchw=inputs_as_nchw
    )
    return onnx_model
