import onnx
#onnx op=10 dan sonra spatial=0 desteğini kaldırdığı için bu değeri 1 olarak değiştiriyoruz.
model = onnx.load(r'arcface_mxnet\resnet100.onnx')

for node in model.graph.node:
    if(node.op_type == "BatchNormalization"):
        for attr in node.attribute:
            if (attr.name == "spatial"):
                attr.i = 1
                
onnx.save(model, r'updated_resnet100.onnx')