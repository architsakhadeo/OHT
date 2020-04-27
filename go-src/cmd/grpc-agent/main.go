package main

import (
	"log"
	"net"

	"github.com/stellentus/cartpoles/go-src/lib/example"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/remote"

	"google.golang.org/grpc"
)

func main() {
	l, err := net.Listen("tcp", ":8081")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	debugLogger := logger.NewDebug(logger.DebugConfig{ShouldPrintDebug: true})
	agent, err := example.NewAgent(debugLogger)
	if err != nil {
		log.Fatalf("failed to init agent: %v", err)
	}

	srv := grpc.NewServer()
	remote.RegisterAgentServer(srv, remote.NewAgentServer(agent))
	srv.Serve(l)
}
