import argparse

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data_path', type=str, required=False, default='../PP_data/')
    #parser.add_argument("--wandb", help="use wandb?", default=True, action=argparse.BooleanOptionalAction) #colab har ik python 3.9 :((((
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.set_defaults(wandb=True)

    parser.add_argument('--kfold', dest='kfold', action='store_true')
    parser.add_argument('--no-kfold', dest='kfold', action='store_false')
    parser.set_defaults(kfold=True)

    parser.add_argument('--custom_config', help="skriv config i net.py set_hyperparameters()!!!", dest='custom_config', action='store_true')
    parser.add_argument('--no-custom_config', dest='custom_config', action='store_false')
    parser.set_defaults(custom_config=False)

    parser.add_argument('--sweep', type=str, required=False)

    parser.add_argument('--scheduler', dest='scheduler', action='store_true')
    parser.add_argument('--no-scheduler', dest='scheduler', action='store_false')
    parser.set_defaults(scheduler=True)
    
    parser.add_argument('--model', type=str, help="vælg en fra model_names", required=False)
    parser.add_argument('--batch_size', type=int, required=False)

    # Parse the argument
    args = parser.parse_args()
    return args