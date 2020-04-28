protoc -I . remote.proto --go_out=plugins=grpc:.
protoc -I . remote.proto --python_out=.
