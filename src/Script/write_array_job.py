import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utils.collect_config import Sweeper
from utils.collect_parser import CollectInput

save_in_folder = "tasks_{}.sh"

def control_job(domains, prev_file=1000, line_per_file=1):

    num_run = 30
    # count_start = count
    count = 0
    file = open(save_in_folder.format(int(prev_file)), 'w')

    for env_name in domains:
        sweeper = Sweeper('../Parameters/{}.json'.format(env_name.lower()), "control_param")
        total_comb = sweeper.get_total_combinations()
        for i in range(total_comb):
            for idx in range(num_run):
                file.write("python control.py" +
                           " --domain " + str(env_name) +
                           " --sweeper_idx " + str(i) +
                           " --run_idx " + str(idx) +
                           "\n"
                           )
                count += 1
                if count % line_per_file == 0:
                    file.close()
                    # print(save_in_folder.format(str((count - count_start - 1) // line_in_file + count_start)), " done")
                    print(save_in_folder.format(str(prev_file), " done"))
                    prev_file += 1
                    if (i+1) * (idx+1) * (domains.index(env_name)+1) < total_comb * num_run * len(domains):
                        # file = open(save_in_folder.format(int((count-count_start) // line_in_file + count_start)), 'w')
                        file = open(save_in_folder.format(str(prev_file)), 'w')
                        print("open new file number", prev_file)
    if not file.closed:
        file.close()


if __name__ == '__main__':
    ci = CollectInput()
    parsers = ci.write_jobs()

    # check if arguments are valid
    domain_list = parsers.domain.split(',')
    line_per_file = int(parsers.lines)

    # write jobs
    if parsers.sweeper == 'control':
        control_job(domain_list, line_per_file=line_per_file)
