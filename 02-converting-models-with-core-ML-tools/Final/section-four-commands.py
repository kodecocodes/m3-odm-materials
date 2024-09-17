import coremltools as ct
import coremltools.optimize as cto

orig_model = ct.models.MLModel("resnet101.mlpackage")

op_config = cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", weight_threshold=512
)
config = cto.coreml.OptimizationConfig(global_config=op_config)
compressed_8_bit_model = cto.coreml.linear_quantize_weights(model, config=config)

compressed_8_bit_model.save("resnet101-8.mlpackage")

from ultralytics import YOLO
model = YOLO("yolov8x-oiv7.pt")
model.export(format="coreml", nms=True, int8=True)
