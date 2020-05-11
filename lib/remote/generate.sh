# This script is used to re-generate the remote.pb.go and python files when remote.proto is updated.
protoc -I . remote.proto --go_out=plugins=grpc:.
python3 -m grpc_tools.protoc -I . remote.proto --python_out=. --grpc_python_out=.
