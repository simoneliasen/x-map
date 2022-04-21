import argparse

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data_path', type=str, required=False, default='../PP_data/')
    parser.add_argument("--wandb", help="use wandb?", default=True, type=bool)
    parser.add_argument("--sweep", help="run sweep on hyperparameters for tuning", default=False, type=bool)
    # Parse the argument
    args = parser.parse_args()
    return args