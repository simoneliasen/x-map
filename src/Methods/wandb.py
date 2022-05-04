import wandb
from copy import deepcopy
from Methods.parser import get_arguments
args = get_arguments()

sweep_config = {
    'name': 'navn',
    'method': 'bayes', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_acc',
    'goal': 'maximize'   
        },
    'parameters': {
        'batch_size': { 
            'values': [32, 64, 128]
        },
        'optimizer': {
            'values': ['sgd', 'rmsprop']
        },
        'exponential_scheduler': { 
            'min': 0.01,
            'max': 0.05, 
        },
        'lr': { 
            'min': 0.001,
            'max': 0.1,
        },
        'weight_decay': { 
            'min': 0.000005,
            'max': 0.0001,
        },
        'dropout_rate': {
            'min': 0.0,
            'max': 0.5,
        },
    }
}

sweep_config['name'] = args.model

if args.sweep is not None: #ellers laver den en ny empty sweep med det navn på wandb dashboard
    del sweep_config['name']

if args.model in ["resnext", "densenet"]:
    print('speciel setting for: ', args.model)
    sweep_config['parameters']['batch_size']['values'] = [32, 64] #resnext og desne kan ik tage > 64
    del sweep_config['parameters']['dropout_rate']['max']#dense og resnext har ikke dropout
    del sweep_config['parameters']['dropout_rate']['min']
    sweep_config['parameters']['dropout_rate']['values'] = [0]

if args.model in ["vgg"]:
    print('speciel setting for: ', args.model)
    sweep_config['parameters']['batch_size']['values'] = [32, 64]

_net = None

def wandb_initialize(net):
    global _net
    _net = net
    sweep_id = wandb.sweep(sweep_config, project="my-test-project", entity="thebigyesman") #todo: dette laver en ny sweep.
    sweep_id2 = sweep_id if args.sweep is None else args.sweep
    wandb.agent(sweep_id=sweep_id2, function=sweep)
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)

def wandb_log(train_loss, val_loss, train_acc, val_acc):
    wandb.log({"train_loss": train_loss})
    wandb.log({"val_loss": val_loss})
    wandb.log({"train_acc": train_acc})
    wandb.log({"val_acc": val_acc})

def wandb_log_folds_avg(avg_val_acc, avg_val_loss):
    wandb.log({"avg_val_acc":avg_val_acc})
    wandb.log({"avg_val_loss":avg_val_loss})

def sweep():
    wandb.init(config=sweep_config)
    net_copy = deepcopy(_net)
    net_copy.set_hyperparameters(wandb.config)
    from Methods.Train import KfoldTrain
    KfoldTrain(net_copy)