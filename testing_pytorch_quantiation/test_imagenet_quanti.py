import sys
sys.path.append("..") 
import os
import copy
from fusing_layers import *
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
# add a random seed so that our results would be reproducable
torch.manual_seed(280012)



directory_data = '/home/sbajaj/MyCode/quantization_remote/Datasets/imagenet_tiny'

transforming_hymen_data = {

    'train_mini': transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val_mini': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),

  }

datasets_images = {x: datasets.ImageFolder(os.path.join(directory_data, x),
                                           transforming_hymen_data[x])
                    for x in ['train_mini', 'val_mini']}
loaders_data = {x: torch.utils.data.DataLoader(datasets_images[x], batch_size=4,
                                             shuffle=True, num_workers=4)
               for x in ['train_mini', 'val_mini']}
sizes_datasets = {x: len(datasets_images[x]) for x in ['train_mini', 'val_mini']}
# print(sizes_datasets)
class_names = datasets_images['train_mini'].classes
n_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = {}".format(device))

def model_training(res_model, criterion, optimizer, scheduler, number_epochs=25):
    since = time.time()
    best_resmodel_wts = copy.deepcopy(res_model.state_dict())
    best_accuracy = 0.0
    for epochs in range(number_epochs):
        print('Epoch {}/{}'.format(epochs, number_epochs - 1))
        print('-' * 10)
        for phase in ['train_mini', 'val_mini']: ## Here each epoch is having a training and validation phase

            if phase == 'train_mini':
               res_model.train()  ## Here we are setting our model to training mode
            else:
               res_model.eval()   ## Here we are setting our model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loaders_data[phase]: ## Iterating over data.
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad() ## here we are making the gradients to zero

                with torch.set_grad_enabled(phase == 'train_mini'): ## forwarding and then tracking the history if only in train
                    outputs = res_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train': # backward and then optimizing only if it is in training phase
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / sizes_datasets[phase]
            epoch_acc = running_corrects.double() / sizes_datasets[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_accuracy: ## deep copy the model
                best_accuracy = epoch_acc
                best_resmodel_wts = copy.deepcopy(res_model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accuracy))

    # load best model weights
    res_model.load_state_dict(best_resmodel_wts)
    return res_model
def evaluate(res_model, phase):
    device = torch.device("cpu")
    res_model.to(device)
    res_model.eval()
    running_corrects = 0

    # count = 0
    for inputs, labels in tqdm(loaders_data[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        since = time.time()
        outputs = res_model(inputs)
        time_elapsed = time.time() - since
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        # count += 1
        # if count > 10:
        #     break
    acc = running_corrects.double() / sizes_datasets[phase]
    print("Acc = {}, time_elapsed = {}".format(acc, time_elapsed))

    return acc, time_elapsed
# Load the model
float_model = resnet18_quantizable(pretrained=True).to('cpu')

# finetune_model = models.resnet18(pretrained=True)
num_ftrs = float_model.fc.in_features
float_model.fc = nn.Linear(num_ftrs, n_classes)
# float_model.fc = nn.Linear(512, n_classes, True)
finetune_model = float_model.to(device)
criterion = nn.CrossEntropyLoss()
finetune_optim = optim.SGD(finetune_model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(finetune_model.parameters(), 0.1)
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

finetune_model = model_training(finetune_model, criterion, finetune_optim, exp_lr_scheduler,
                        number_epochs=25)


# Our initial baseline model which is FP32
model_fp32 = finetune_model
model_fp32.eval()

# Sets the backend for x86
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepares the model for the next step i.e. calibration.
# Inserts observers in the model that will observe the activation tensors during calibration
model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False)

acc_fp32, time_elapsed_fp32 = evaluate(model_fp32_prepared, "val_mini")

# Converts the model to a quantized model(int8) 
model_quantized = torch.quantization.convert(model_fp32_prepared) # Quantize the model

acc_int8, time_elapsed_int8 = evaluate(model_quantized, "val_mini")


print("accuracy loss = {}".format((acc_fp32 - acc_int8)/acc_fp32))
print("speed up = {}".format(time_elapsed_fp32/time_elapsed_int8))