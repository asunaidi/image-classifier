import json
import torch
import copy
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import argparse
import utility

parser = argparse.ArgumentParser(description='Train a Neural Network on Your flowers Dataset')


parser.add_argument('--image_path',
                    type=str,
                    help="the path of the flower image you want to predict",
                    required = True,
                    default='flowers/test/52/image_04221.jpg')
parser.add_argument('--checkpoint',
                    type=str,
                    help="checkpoint file name",
                    required = True,
                    default='checkpoint.pth')
parser.add_argument('--top_k',
                    type=int,
                    help="the number of top classes you want to view the probability of",
                    required = False,
                    default=5)
parser.add_argument('--cat_map',
                type = str,
                help = 'the name of the json file that maps each category to a class',
                required = False,
                default='cat_to_name.json')
parser.add_argument('--gpu',
                type = bool,
                nargs='?',
                const=True,
                help = 'Whether you want your model to be trained on GPU',
                required = False,
                default=0)


args = parser.parse_args()
image_path = args.image_path
checkpoint_file = args.checkpoint
top_k = args.top_k
cat_map_file = args.cat_map
gpu_mode = args.gpu

print(args)

print('predicting')


with open(cat_map_file, 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda" if (torch.cuda.is_available() and gpu_mode) else "cpu")

checkpoint_dir = 'checkpoints/'
model_path = checkpoint_dir + checkpoint_file
model = utility.load_model(model_path, device)
utility.sanity_check(image_path, model, top_k, device, cat_to_name)
