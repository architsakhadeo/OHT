import os
from utils.collect_config import ParameterConfig
import pickle as pkl

def write_param_log(agent_params, env_params, env, file_path, exp_params=None, save_pkl=False):
    if os.path.isfile(file_path + "/record.pkl"):
        print("Log exist")

    if save_pkl:
        obj = ParameterConfig()
        setattr(obj, "agent_params", agent_params)
        setattr(obj, "env_params", env_params)
        setattr(obj, "environment", env)
        with open(file_path+"/record.pkl", "wb") as param_obj:
            pkl.dump(obj, param_obj)
        print("param saved in", file_path)

    with open(file_path + "/param.txt", "w") as param_record:
        param_record.write("------ Agent parameters ------\n\n")
        est_len = 20
        for pair in agent_params.__dict__:
            space = " " * (est_len - len(str(pair))) + ": "
            print(str(pair), space, str(agent_params.__dict__[pair]))
            info = str(pair) + space + str(agent_params.__dict__[pair]) + "\n"
            param_record.write(info)
        param_record.write("\n\n------ Environment parameters ------\n\n")
        param_record.write("Env: " + str(env) + "\n\n")
        for pair in env_params.__dict__:
            space = " " * (est_len - len(str(pair))) + ": "
            print(str(pair), space, str(env_params.__dict__[pair]))
            info = str(pair) + space + str(env_params.__dict__[pair]) + "\n"
            param_record.write(info)
        if exp_params is not None:
            param_record.write("\n\n------ Control exp parameters ------\n\n")
            for pair in exp_params.__dict__:
                space = " " * (est_len - len(str(pair))) + ": "
                print(str(pair), space, str(exp_params.__dict__[pair]))
                info = str(pair) + space + str(exp_params.__dict__[pair]) + "\n"
                param_record.write(info)

    print("log saved in", file_path)

