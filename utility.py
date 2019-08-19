import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import argparse
from PIL import Image


def train_model(model, n_epochs, criterion, optimizer, scheduler, device, dataloaders):

    model.to(device)
    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        print('new epoch')
        training_loss = 0.0
        validation_loss = 0.0
        training_accuracy = 0.0
        validation_accuracy =0.0
        
        scheduler.step()
        model.train()
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device) 
            
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            training_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        training_loss = training_loss / len(dataloaders['train'])
        training_accuracy = training_accuracy / len(dataloaders['train'])

            
            
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs = inputs.to(device)
                labels = labels.to(device) 
                output = model.forward(inputs)
                loss = criterion(output, labels)
                validation_loss += loss.item()
            
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        validation_loss = validation_loss/len(dataloaders['valid'])
        validation_accuracy = validation_accuracy/len(dataloaders['valid'])
        
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_model = copy.deepcopy(model.state_dict())
            
        print(f"Epoch {epoch+1}/{n_epochs}.. "
              f"Training Loss: {training_loss:.3f}.. "
              f"Training Accuracy: {training_accuracy*100:.3f}%.."
              f"Validation Loss: {validation_loss:.3f}.. "
              f"Validation Accuracy: {validation_accuracy*100:.3f}%")    
                

 
    print(f"Best Accuracy achieved: {best_accuracy*100:.3f}%")

    model.load_state_dict(best_model)
    return model

def load_model(model_path, device):
    checkpoint = torch.load(model_path)
    if checkpoint['arch'] =='vgg16':
        model = models.vgg16(pretrained=True)
        n_features = 25088
    elif checkpoint['arch'] == 'densnet121':
        model = models.densnet121(pretrained=True)
        n_features == 1024
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    model.classifier = nn.Sequential(nn.Linear(n_features, hidden_units),nn.ReLU(),nn.Dropout(0.2),nn.Linear(hidden_units, 50),nn.ReLU(),nn.Linear(50, 102),nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):

    image = Image.open(image_path)
    if image.size[0] < image.size[1]:
        image.thumbnail((256, 10000),Image.ANTIALIAS)
    else:
        image.thumbnail((10000, 256),Image.ANTIALIAS)
        
    left_cor = (image.size[0]-224)/2
    bottom_cor = (image.size[1]-224)/2

    image = image.crop((left_cor, bottom_cor, (left_cor + 224), (bottom_cor + 224)))
    
    image = np.array(image)/255
    image = (image - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225]) 
    image = image.transpose((2, 0, 1))
    return image  



def predict(image_path, model, topk, device):

    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    image = image.unsqueeze(0)
    image.to(device)
    
    output = model.forward(image)
    probs = torch.exp(output) 
    top_ps, top_classes = probs.topk(topk)
    
    top_ps = top_ps.detach().numpy().tolist()[0] 
    top_classes = top_classes.detach().numpy().tolist()[0]
    
    indices_to_classes = {index: label for label, index in    
                                      model.class_to_idx.items()}
        
    top_labels = [indices_to_classes[index] for index in top_classes]
    
    return top_ps, top_labels

def sanity_check(image_path, model, top_k, device, cat_to_name):
    top_probs, top_classes = predict(image_path, model, top_k, device)
    top_flowers = {cat_to_name[label] for label in top_classes}

    flower_class_num = image_path.split('/')[2]
    flower_class_name = cat_to_name[flower_class_num]
    print('The real flower name:' + flower_class_name)
    print('Your model predictions:')
    top_probs = list(top_probs)
    top_flowers = list(top_flowers)
    for x in range(top_k):
        print(f"{x+1}: {top_flowers[x]}.. prob : {top_probs[x]} ")
