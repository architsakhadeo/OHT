# cartpoles

Cartpole agents and environments for real-world RL.

## Install Instructions

Unless otherwise noted, these commands are executed in the `cartpoles` directory.

### Initial Setup

1. Run `python3 -m venv env` (or `py -m venv env` on Windows). This will create the virtual environment. This only needs to be done once.
2. Execute `source env/bin/activate` (or `.\env\Scripts\activate` on Windows) to enter the virtual environment.
3. Execute `pip install -r requirements.txt` to install required packages.
4. Execute `deactivate` to exit the virtual environment.

### Development Workflow

1. Execute `source env/bin/activate` (or `.\env\Scripts\activate` on Windows) to enter the virtual environment.
2. Change into the Experiments directory: `cd python/Experiments/`. The control script currently only works from that directory. Then run the code with `python control.py`. Parameters can be changed by providing a different parameters JSON file.
3. If you need a new package, install it in the virtual environment with `python -m pip install PACKAGE_NAME`.
4. When you're done working on the project, `deactivate` exits the virtual environment.
