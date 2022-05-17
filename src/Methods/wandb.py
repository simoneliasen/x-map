import wandb
from copy import deepcopy
from Methods.parser import get_arguments
args = get_arguments()

sweep_config_weight_decay_test = { #til weight decay test
    'name': 'navn', #alle andre er bare sat til den hypertunede.
    'method': 'grid', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_loss',
    'goal': 'minimize'   
        },
    'parameters': {
        'batch_size': { 
            'values': [32]
        },
        'exponential_scheduler': { 
            'values': [0.028372597647743948]
        },
        'lr': { 
            'values': [0.002987017391774973]
        },
        'weight_decay': { #1x, 10x, 100x, 1000x, 10.000x
            'values': [0.00009916691342646931, 0.001, 0.01, 0.1, 1]
        },
        'dropout_rate': {
            'values': [0.05233710200284458]
        },
    }
}

sweep_config = {
    'name': 'navn',
    'method': 'random', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_loss',
    'goal': 'minimize'   
        },
    'parameters': {
        'batch_size': { 
            'values': [32, 64]
        },
        'exponential_scheduler': { 
            'min': 0.025,
            'max': 0.04, 
        },
        'lr': { 
            'distribution': 'log_uniform',
            'min': -4, #log(0.0001)
            'max': -1.39794000867, #log(0.04)
            #LOG UNIFORM!!!
        },
        'weight_decay': {
            'distribution': 'log_uniform', 
            'min': -3, #log 0.001
            'max': -1, #log 0.1
            #LOG UNIFORM!!! 10x - 1000x
        },
        'dropout_rate': {
            'min': 0.0,
            'max': 0.3,
        },
    }
}

sweep_config_data_augmentation = { #TIL DATA AUGMENTATION
    'name': 'navn',
    'method': 'grid', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_loss',
    'goal': 'minimize'   
        },
    'parameters': {
        'vertical_flip': {
            'values': ['true', 'false']
        },
        'horizontal_flip': { 
            'values': ['true', 'false']
        },
        'random_affine': { 
            'values': ['true', 'false']
        },
        'color_jitter': { 
            'values': ['true', 'false']
        },
        'grayscale': { 
            'values': ['true', 'false']
        },
    }
}

sweep_config_old = {
    'name': 'navn',
    'method': 'bayes', #grid, random, bayesian
    'metric': {
    'name': 'avg_val_loss',
    'goal': 'minimize'   
        },
    'parameters': {
        'batch_size': { 
            'values': [32, 64]
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

def load_model_sweep_configs():
    sweep_config['name'] = f"{args.model}_version_{args.model_version}"

    if args.baseline:
        sweep_config['name'] = 'Baseline'

    #ellers laver den en ny empty sweep med det navn på wandb dashboard
    if args.sweep is not None: 
        del sweep_config['name']
    
    if args.name != 'none':
        sweep_config['name'] = args.name

    #dense og resnext har ikke dropout
    if args.model in ["resnext", "densenet"]:
        print('ingen dropout for: ', args.model)
        del sweep_config['parameters']['dropout_rate']

    #128 i max batch
    if args.model in ["inception"]:
        sweep_config['parameters']['batch_size']['values'] = [32, 64, 128]

    if args.model == "efficientnet" and args.model_version == 0:
        sweep_config['parameters']['batch_size']['values'] = [32, 64, 128]

    #32 i maxbatch size
    if args.model == "resnext" and args.model_version == 1:
        sweep_config['parameters']['batch_size']['values'] = [32]

    if args.model == "densenet" and args.model_version == 1:
        sweep_config['parameters']['batch_size']['values'] = [32]

    if args.model == "densenet" and args.model_version == 2:
        sweep_config['parameters']['batch_size']['values'] = [32]


    #16 i max batch
    if args.model == "efficientnet" and args.model_version == 1:
        sweep_config['parameters']['batch_size']['values'] = [16]


    #2 i max batch
    if args.model == "efficientnet" and args.model_version == 2:
        sweep_config['parameters']['batch_size']['values'] = [2]


load_model_sweep_configs()

_net = None

def wandb_initialize(net):
    global _net
    _net = net
    sweep_id = wandb.sweep(sweep_config, project="my-test-project", entity="thebigyesman") #todo: dette laver en ny sweep.
    sweep_id2 = sweep_id if args.sweep is None else args.sweep
    wandb.agent(sweep_id=sweep_id2, function=sweep)
    #kan også bruge et specifikt sweep_id, fx f7pvbfd4 (find på wandb under sweeps)
    #wandb.watch(model)

def wandb_log(type, loss, acc, sensitivity, precision, specificity, FalseNegativeRate, FalsePositiveRate):
    if type not in ["train", "val", "test"]:
        print("FORKERT LOGGET DUDE!")
    else:
        wandb.log({f"{type}_loss": loss})
        wandb.log({f"{type}_acc": acc})
        wandb.log({f"{type}_sensitivity": sensitivity})
        wandb.log({f"{type}_precision": precision})
        wandb.log({f"{type}_specificity": specificity})
        wandb.log({f"{type}_falseNegativeRate": FalseNegativeRate})
        wandb.log({f"{type}_falsePositiveRate": FalsePositiveRate})



def wandb_log_folds_avg(avg_val_acc, avg_val_loss):
    wandb.log({"avg_val_acc":avg_val_acc})
    wandb.log({"avg_val_loss":avg_val_loss})

def sweep():
    wandb.init(config=sweep_config)
    net_copy = deepcopy(_net)
    if args.baseline:
        net_copy.set_baselineparameters()
    else:
        net_copy.set_hyperparameters(wandb.config) #sæt for baseline, vores, og wandb?

    net_copy.set_transforms(wandb.config)
    from Methods.Train import KfoldTrain
    KfoldTrain(net_copy)