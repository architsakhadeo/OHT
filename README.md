# cartpoles

Cartpole agents and environments for real-world RL.

## Running Experiments

1. `go build cmd/experiment/main.go` This provides a slight speedup over using `go run` in the next step (about 3 seconds per run).
2. `time parallel './main -config "config/esarsa.json" -run' ::: {1..60}` This runs 60 experiments in parallel. They'll all use the default `sweep=0`, but a different run number. Note this uses GNU Parallel.
3. Change `config/esarsa.json` to a different config as necessary. (As much as possible it's good to run experiments based on versions of a config that have been committed. Then we have a history and can re-run previous versions.)

## Plotting with Jupyter

1. Make sure you're in the virtualenv: `source env/bin/activate`.
2. If you haven't yet done so, install Jupyter from the pip file: `pip install -r requirements.txt`.
3. Launch Jupyter: `jupyter notebook --notebook-dir="plot"`. (The argument isn't necessary. It just opens the plot folder directly.)

## Development Workflow

1. Execute `source env/bin/activate` (or `.\env\Scripts\activate` on Windows) to enter the virtual environment.
2. Change into the Experiments directory: `cd python/Experiments/`. The control script currently only works from that directory. Then run the code with `python control.py`. Parameters can be changed by providing a different parameters JSON file.
3. If you need a new package, install it in the virtual environment with `python -m pip install PACKAGE_NAME` and add it to the requirements with `pip freeze > requirements.txt`.
4. When you're done working on the project, `deactivate` exits the virtual environment.

## Install Instructions

1. Run `python3 -m venv env` (or `py -m venv env` on Windows). This will create the virtual environment. This only needs to be done once.
2. Execute `source env/bin/activate` (or `.\env\Scripts\activate` on Windows) to enter the virtual environment.
3. Execute `pip install --upgrade pip` to upgrade pip.
3. Execute `pip install -r requirements.txt` to install required packages.
4. Execute `deactivate` to exit the virtual environment.

### Protocol Buffers

*This is only necessary if you need to modify the protocol buffer definitions (which is unlikely).*

* Install protocol buffers. I downloaded the C++ code and compiled from source. These instructions might work for you: https://github.com/protocolbuffers/protobuf/blob/master/src/README.md
* Install the go plugin: `go get -u github.com/golang/protobuf/protoc-gen-go`
* From within an active `virtualenv` session, install the python plugin: `python -m pip install grpcio`
* Now you can execute `lib/remote/generate.sh`
