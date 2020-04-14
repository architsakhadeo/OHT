Usage: 
Run the following command in Experiment folder:
python control.py --domain CCP --sweeper_idx 0 --run_idx 0

Parameters in json:
learning: 
    "offline" - learn a policy from a given (handcoded) controller. Then evaluate the offline agent  
    "online" - learn a policy online

agent:
    The agent name for learning

learn_from:
    The name of the given cotroller. It is only used in offline mode.
    
rep_type:
    "sepTC": tile coding each bit separately
    "obs": normalized raw observation