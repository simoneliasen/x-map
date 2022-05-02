import wandb
from Methods.parser import get_arguments
args = get_arguments()

sweep_config = {
    'name': 'resnext',
    'method': 'bayes', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_acc',
    'goal': 'maximize'   
        },
    'parameters': {
        'batch_size': { 
            #'values': [32, 64, 128] 
            'values': [32, 64] #resnext og desne kan ik tage > 64
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
        'dropout_rate': { #dense og resnext har ikke dropout
            'values': [0]
            #'min': 0.0,
            #'max': 0.5,
        },
    }
}

_net = None

def wandb_initialize(net):
    global _net
    _net = net
    sweep_id = wandb.sweep(sweep_config, project="my-test-project", entity="thebigyesman")
    sweep_id2 = sweep_id if args.sweep is None else args.sweep
    wandb.agent(sweep_id=sweep_id2, function=sweep)
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)

def wandb_log(train_loss, val_loss, train_acc, val_acc):
    wandb.log({"train_loss": train_loss})
    wandb.log({"val_loss": val_loss})
    wandb.log({"train_acc": train_acc})
    wandb.log({"val_acc": val_acc})

def wandb_log_folds_avg(avg_val_acc):
    wandb.log({"avg_val_acc":avg_val_acc})

def sweep():
    wandb.init(config=sweep_config)
    _net.set_hyperparameters(wandb.config)
    from Methods.Train import KfoldTrain
    KfoldTrain(_net)