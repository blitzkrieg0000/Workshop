python -m grpc_tools.protoc -I./proto --python_out=./proto/ --grpc_python_out=./proto/ ./proto/sample.proto
#python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. ManagerService.proto

