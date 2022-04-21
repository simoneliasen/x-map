import wandb

sweep_config = {
    'method': 'random', #grid, random, bayesian
    'metric': {
    'name': 'test_acc',
    'goal': 'maximize'   
        },
    'parameters': {
        'batch_size': { 
            'values': [2, 4, 3] 
        },
        'lr': { 
            'values': [0.000005, 0.00005, 0.0005]
        },
    }
}

_net = None

def wandb_initialize(net):
    global _net
    _net = net
    sweep_id = wandb.sweep(sweep_config, project="my-test-project", entity="thebigyesman")
    wandb.agent('f7pvbfd4', function=sweep)

    #wandb.init(project="my-test-project", entity="thebigyesman")

    #wandb.config = {
    #"learning_rate": 0.001,
    #"epochs": 100,
    #"batch_size": 128
    #}

    #wandb.watch(model)

def wandb_log(train_loss, test_loss, train_acc, test_acc):
    wandb.log({"train_loss": train_loss})
    wandb.log({"test_loss": test_loss})
    wandb.log({"train_acc": train_acc})
    wandb.log({"test_acc": test_acc})

def sweep():
    wandb.init(config=sweep_config)
    parameters = dict()
    parameters['batch_size'] = wandb.config['batch_size']
    parameters['lr'] = wandb.config['lr'] #try same for both
    _net.set_hyperparameters(parameters)
    from Methods.Train import KfoldTrain
    KfoldTrain(_net)