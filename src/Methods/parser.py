import argparse

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data_path', type=str, required=False, default='../PP_data/')
    #parser.add_argument("--wandb", help="use wandb?", default=True, action=argparse.BooleanOptionalAction) #colab har ik python 3.9 :((((
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no-wandb', action='store_false')
    parser.set_defaults(wandb=True)
    parser.add_argument("--sweep", help="run sweep on hyperparameters for tuning", default=False, action=argparse.BooleanOptionalAction)
    # Parse the argument
    args = parser.parse_args()
    return args