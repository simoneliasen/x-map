import argparse

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--data_path', type=str, required=False, default='../PP_data/')
    #parser.add_argument("--wandb", help="use wandb?", default=True, action=argparse.BooleanOptionalAction) #colab har ik python 3.9 :((((
    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.set_defaults(wandb=False)

    parser.add_argument('--kfold', dest='kfold', action='store_true')
    parser.add_argument('--no-kfold', dest='kfold', action='store_false')
    parser.set_defaults(kfold=True)

    parser.add_argument('--custom_config', help="skriv config i net.py set_hyperparameters()!!!", dest='custom_config', action='store_true')
    parser.add_argument('--no-custom_config', dest='custom_config', action='store_false')
    parser.set_defaults(custom_config=False)

    parser.add_argument('--baseline', help="skriv config i net.py set_hyperparameters()!!!", dest='baseline', action='store_true')
    parser.add_argument('--no-baseline', dest='baseline', action='store_false')
    parser.set_defaults(baseline=False)

    parser.add_argument('--sweep', type=str, required=False)

    parser.add_argument('--scheduler', dest='scheduler', action='store_true')
    parser.add_argument('--no-scheduler', dest='scheduler', action='store_false')
    parser.set_defaults(scheduler=True)
    
    parser.add_argument('--model', type=str, help="vælg en fra model_names", required=False, default="vgg") #vgg fordi den vandt. Og baseline er same.
    parser.add_argument('--name', type=str, help="navn til dit sweep", required=False, default='none')
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--model_version', type=int, required=False, default=2) #default 2 fordi vgg19 vandt. Samme version for baseline.

    # Parse the argument
    args = parser.parse_args()
    return args