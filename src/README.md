Usage: 
Run the following command in Experiment folder:

run control experiment:
    python control.py --json CCP --sweeper_idx 0 --run_idx 0

plotting code (You may need to modify plotting code or write new functions):
    python plot.py 

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