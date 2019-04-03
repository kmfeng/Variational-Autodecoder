import sys
import json
import itertools
import os
import csv
import random

config_folder = sys.argv[1]

# valid values: ints
n_latent_params = [25, 50, 100]
hidden = [50, 100, 200]
n_layers = [2, 4, 6]

# true or false
use_adam = [True]

# floats
latent_param_lr = [0.001, 0.0001]
net_lr = [0.001, 0.0001]
test_latent_param_lr = [0.001, 0.0001]

# None, 0.1 to 0.9
# masked = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
masked = [None, 0.1, 0.2, 0.3]

dropout = [None]
layer_norm = [False]

latent_init = ['pca']
test_latent_init = ['normal']

activation_fn = ['relu']

options = [
    ('n_latent_params', n_latent_params),
    ('n_hidden', hidden),
    ('n_hidden_layers', n_layers),
    ('use_adam', use_adam),
    ('dropout', dropout),
    ('layer_norm', layer_norm),
    ('latent_param_lr', latent_param_lr),
    ('net_lr', net_lr),
    ('test_latent_param_lr', test_latent_param_lr),
    ('masked', masked),
    ('latent_param_init', latent_init),
    ('test_latent_param_init', test_latent_init),
    ('activation_fn', activation_fn),
]

option_names = [x[0] for x in options]
option_values = [x[1] for x in options]

extra_fields = [
    'config_folder',
    'config_num',
]

basedir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(basedir, config_folder)

print('writing to {}'.format(config_path))

# find last config file
if not os.path.exists(config_path):
    os.mkdir(config_path)

files = os.listdir(config_path)
files = [x for x in files if x.endswith(".json") and x.startswith("config")]

first_file = len(files)
print(first_file)

g = open(os.path.join(basedir, 'config_list_{}.csv'.format(config_folder)), 'w')
writer = csv.writer(g)
writer = csv.DictWriter(g, fieldnames=extra_fields+option_names)
writer.writeheader()

combis = [combi for combi in itertools.product(*option_values)]
print("# configs:", len(combis))
random.shuffle(combis)
# print(combis[:2])
for i, combi in enumerate(combis):
    config_num = i + first_file
    filename = os.path.join(config_path, "config_{}.json".format(config_num))
    if os.path.exists(filename):
        print("Error! file {} already exists!".format(filename))

    ops = {option_names[j]: combi[j] for j in range(len(options))}
    ops['config_folder'] = config_folder
    ops['config_num'] = config_num

    with open(filename, 'w') as f:
        json.dump(ops, f, indent=4)

    writer.writerow(ops)

g.close()

print('saved {} config files to {}'.format(i+1, config_path))
