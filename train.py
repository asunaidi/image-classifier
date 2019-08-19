import json
import torch
import copy
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import argparse
import utility


parser = argparse.ArgumentParser(description='Train a Neural Network on Your flowers Dataset')


parser.add_argument('--data_dir',
                    type =str,
                    help='the folder of the data you want to train',
                    required=True,
                    default='flowers')
parser.add_argument('--checkpoint',
                    type=str,
                    help="checkpoint file name",
                    required = False,
                    default="checkpoint.pth")
parser.add_argument('--save_dir', 
                    type=str,
                    help='the folder where you want to store your models',
                    required = False,
                    default='checkpoints/')
parser.add_argument('--learning_rate',
                    type=float,
                    help='learning rate',
                    required = False,
                    default=0.001)
parser.add_argument('--hidden_units',
                    type=int,
                    help='number of hidden units',
                    required = False,
                    default = 150)                    
parser.add_argument('--epochs',
                type =int,
                help='the number of times you want to adjust your model weights',
                required = False,
                default=12)
parser.add_argument('--arch',
                type = str,
                required = False,
                help = 'The architecture of the Neural Network pre-trainded model used for extracting features',
                default='vgg16')
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
                default=1)


args = parser.parse_args()
data_dir = args.data_dir
checkpoint_file = args.checkpoint
save_dir = args.save_dir
learning_rate = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
cat_map_file = args.cat_map
gpu_mode = args.gpu
n_epochs = args.epochs

print(args)


print('training')
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    'valid' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    'test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
}

dirs = {'train': train_dir, 'valid': valid_dir, 'test' : test_dir}
image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}

with open(cat_map_file, 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda" if (torch.cuda.is_available() and gpu_mode) else "cpu")

if architecture =='vgg16':
    model = models.vgg16(pretrained=True)
    n_features = 25088
elif architecture == 'densnet121':
    model = models.densnet121(pretrained=True)
    n_features == 1024
           
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(n_features, hidden_units),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden_units, 50),nn.ReLU(),nn.Linear(50, 102),nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
print('starting')
model = utility.train_model(model, n_epochs, criterion, optimizer, scheduler, device, dataloaders)

model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
checkpoint  = { 'arch': architecture,
                'hidden_units': hidden_units,
                'class_to_idx': model.class_to_idx, 
                'state_dict': model.state_dict()}
checkpoint_file_path =  save_dir + checkpoint_file
torch.save(checkpoint, checkpoint_file_path)


print("Congrats! Your model has been successfully trained") 