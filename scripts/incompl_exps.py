def incompl_exp(name='cartpole', total=120):
    all_exp = set(list(range(total)))

    with open(f'{name}_completed') as f:
        comp = f.readlines()

    comp_exp = set()
    for line in comp:
        if 'target' in line or line.strip() == '':
            continue
        line_num = line.split('/')[-3].split('_')[-1]
        comp_exp.add(int(line_num))

    incomp_exp = all_exp.difference(comp_exp)
    if len(incomp_exp) == 0:
        return
    print(f"{name}: len={len(incomp_exp)}\n{incomp_exp}")

    with open(f'{name}_incompleted', 'w') as f:
        for i in incomp_exp:
            f.write(f'{i}\n')



if __name__ == "__main__":
    # incompl_exp(name='cartpole')
    # Offline training with NN
    # incompl_exp(name='acrobot', total=300)
    # incompl_exp(name='puddle', total=300)

    # Offline training with TC
    # incompl_exp(name='acrobot_linear', total=150)
    # incompl_exp(name='puddle_linear', total=150)

    # offline NN, no hyperparam sweep
    incompl_exp(name='acrobot_step5k_lambda1e-1', total=30)
    incompl_exp(name='acrobot_step5k_lambda1e-3', total=30)
    incompl_exp(name='acrobot_step5k_lambda1e-5', total=30)
    incompl_exp(name='acrobot_step15k_lambda1e-1', total=30)
    incompl_exp(name='acrobot_step15k_lambda1e-3', total=30)
    incompl_exp(name='acrobot_step15k_lambda1e-5', total=30)

    incompl_exp(name='puddlerand_step5k_lambda1e-1', total=30)
    incompl_exp(name='puddlerand_step5k_lambda1e-3', total=30)
    incompl_exp(name='puddlerand_step5k_lambda1e-5', total=30)
    incompl_exp(name='puddlerand_step15k_lambda1e-1', total=30)
    incompl_exp(name='puddlerand_step15k_lambda1e-3', total=30)
    incompl_exp(name='puddlerand_step15k_lambda1e-5', total=30)


    # offline TC, no hyperparam sweep
    incompl_exp(name='acrobot_linear_step5k_lambda1e-1', total=30)
    incompl_exp(name='acrobot_linear_step5k_lambda1e-3', total=30)
    incompl_exp(name='acrobot_linear_step5k_lambda1e-5', total=30)
    incompl_exp(name='acrobot_linear_step15k_lambda1e-1', total=30)
    incompl_exp(name='acrobot_linear_step15k_lambda1e-3', total=30)
    incompl_exp(name='acrobot_linear_step15k_lambda1e-5', total=30)

    incompl_exp(name='puddlerand_linear_step5k_lambda1e-1', total=30)
    incompl_exp(name='puddlerand_linear_step5k_lambda1e-3', total=30)
    incompl_exp(name='puddlerand_linear_step5k_lambda1e-5', total=30)
    incompl_exp(name='puddlerand_linear_step15k_lambda1e-1', total=30)
    incompl_exp(name='puddlerand_linear_step15k_lambda1e-3', total=30)
    incompl_exp(name='puddlerand_linear_step15k_lambda1e-5', total=30)