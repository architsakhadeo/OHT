import json

class Sweeper(object):
    """
    from https://github.com/muhammadzaheer/classic-control/blob/master/sweeper.py
    """
    def __init__(self, config_file, sweep_param_key):
        with open(config_file) as f:
            self.config_dict = json.load(f)
        self.total_combinations = 1
        self.set_total_combinations(self.config_dict[sweep_param_key])
        self.sweep_param_key = sweep_param_key

    def set_total_combinations(self, sweep_params):
        tc = 1
        for c_name, c_value in sweep_params.items():
            if type(c_value) == dict:
                sub_tc = self.set_total_combinations(c_value)
                tc *= sub_tc
            else:
                tc *= len(c_value)
        self.total_combinations = tc
        return tc

    def get_total_combinations(self):
        return self.total_combinations

    def parse(self, idx):
        assert idx < self.total_combinations, 'idx must be less than total combination of the parameters.'
        config = ParameterConfig()
        set_param = self.define_swept_param(self.config_dict["default_param"], self.config_dict[self.sweep_param_key], idx)
        config.set_attributes(set_param)
        return config

    def define_swept_param(self, default_file, sweep_param_file, idx):
        cumulative = 1
        for c_name, c_value in sweep_param_file.items():
            if type(c_value) == dict:
                sub_dict = self.define_swept_param(sweep_param_file[c_name], c_value, idx)
                default_file[c_name] = {**default_file[c_name], **sub_dict}
            else:
                num_values = len(c_value)
                default_file[c_name] = c_value[int(idx / cumulative) % num_values]
                cumulative *= num_values
        return default_file


class ParameterConfig(object):
    def __init__(self):
        pass

    @staticmethod
    def load_json(file_name):
        json_data = open(file_name, 'r')
        config_file = json.load(json_data)
        json_data.close()
        return config_file

    def set_attributes(self, config_file):
        for c_name, c_value in config_file.items():
            setattr(self, c_name, c_value)
            if type(getattr(self, c_name)) == dict:
                sub_obj = ParameterConfig()
                sub_obj = sub_obj.set_attributes(getattr(self, c_name))
                setattr(self, c_name, sub_obj)
        return self

# # test
# if __name__ == '__main__':
#     sweeper = Sweeper('../Parameters/ww.json')
#     sweeper.parse(0)
