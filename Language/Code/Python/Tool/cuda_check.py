import os
import cv2
import torch
#import dlib
import tensorflow as tf
import onnxruntime  as ort

#Uyarıları gösterme
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

torch_status = torch.cuda.is_available()
print("\ntorch: ", torch_status)

cv2_status = cv2.cuda.getCudaEnabledDeviceCount()
print("cv2: ", True if cv2_status>0 else False, "count: ", cv2_status)

onnxruntime_status = ort.get_device()
print("onnxruntime: ", True if onnxruntime_status=="GPU" else False)

#dlib_status = dlib.DLIB_USE_CUDA
#print("dlib: ", dlib_status)

tensorflow_status = tf.config.list_physical_devices('GPU')[0].device_type
print("tensorflow: ",  True if tensorflow_status=="GPU" else False, end="\n")