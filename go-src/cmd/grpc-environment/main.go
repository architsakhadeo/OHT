package main

import (
	"log"
	"net"

	"github.com/stellentus/cartpoles/go-src/lib/environment"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/remote"

	"google.golang.org/grpc"
)

func main() {
	l, err := net.Listen("tcp", ":8080")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	debugLogger := logger.NewDebug(logger.DebugConfig{ShouldPrintDebug: true})
	env, err := environment.NewExample(debugLogger)
	if err != nil {
		log.Fatalf("failed to init environment: %v", err)
	}

	srv := grpc.NewServer()
	remote.RegisterEnvironmentServer(srv, remote.NewEnvironmentServer(env))
	srv.Serve(l)
}
