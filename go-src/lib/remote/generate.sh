protoc -I . remote.proto --go_out=plugins=grpc:.
python3 -m grpc_tools.protoc -I . remote.proto --python_out=. --grpc_python_out=.
