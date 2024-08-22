import onnx
from onnxconverter_common import float16

model = onnx.load("./pretrained_model/model_1.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "./pretrained_model/model_1_fp16.onnx")