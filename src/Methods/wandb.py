import wandb

sweep_config = {
    'name': 'dense2',
    'method': 'bayes', #grid, random, bayesian
    'metric': {
    'name': 'test_acc',
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

_net = None

def wandb_initialize(net):
    global _net
    _net = net
    sweep_id = wandb.sweep(sweep_config, project="my-test-project", entity="thebigyesman")
    wandb.agent(sweep_id=sweep_id, function=sweep)
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)

def wandb_log(train_loss, test_loss, train_acc, test_acc):
    wandb.log({"train_loss": train_loss})
    wandb.log({"test_loss": test_loss})
    wandb.log({"train_acc": train_acc})
    wandb.log({"test_acc": test_acc})

def sweep():
    wandb.init(config=sweep_config)
    _net.set_hyperparameters(wandb.config)
    from Methods.Train import KfoldTrain
    KfoldTrain(_net)