from tensorflow.keras.models import load_model, save_model
import argparse
import tf2onnx
import onnx

def model2onnx():
  model_path = "/home/blitzkrieg/Desktop/Mask Detection/weights/mask_detector.h5"
  model = load_model(model_path)
  onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

  onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
  onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'
  onnx.save(onnx_model, "/home/blitzkrieg/Desktop/Mask Detection/weights/mask_detector.onnx")

if __name__ == "__main__":
	model2onnx()