package main

import (
	"log"
	"net"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/remote"

	"google.golang.org/grpc"
)

func main() {
	l, err := net.Listen("tcp", ":8081")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	debugLogger := logger.NewDebug(logger.DebugConfig{ShouldPrintDebug: true})
	agent, err := agent.NewExample(debugLogger)
	if err != nil {
		log.Fatalf("failed to init agent: %v", err)
	}

	srv := grpc.NewServer()
	remote.RegisterAgentServer(srv, remote.NewAgentServer(agent))
	srv.Serve(l)
}
