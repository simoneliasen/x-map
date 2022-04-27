import argparse

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data_path', type=str, required=False, default='../PP_data/')
    parser.add_argument("--wandb", help="use wandb?", default=True, action='store_false')
    parser.add_argument("--sweep", help="run sweep on hyperparameters for tuning", default=False, action='store_true')
    # Parse the argument
    args = parser.parse_args()
    return args