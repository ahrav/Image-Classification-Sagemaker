#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, use_cuda):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # Monitor test loss and accuracy.
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    model.eval()
    for inputs, labels in test_loader:
        # Move to GPU.
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs)
        loss = criterion(output, labels)
        _, preds = torch.max(output, 1)
        test_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)

    total_loss = test_loss // len(test_loader)
    total_acc = correct.double() // len(test_loader)

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}\n".format(
            total_loss, total_acc)
    )


def train(model, train_loader, test_loader, criterion, optimizer, use_cuda):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs = 5
    train_loss = 0
    correct_pred = 0

    for e in range(epochs):
        model.train()

        for inputs, labels in train_loader:
            # Move to GPU.
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output, 1)
            train_loss += loss.item() * inputs.size(0)
            correct_pred += torch.sum(preds == labels.data)

        epoch_loss = train_loss // len(train_loader)
        epoch_acc = correct_pred // len(train_loader)

        logger.info("\nEpoch: {}/{}.. ".format(e+1, epochs))
        logger.info("Training Loss: {:.4f}".format(epoch_loss))
        logger.info("Training Accuracy: {:.4f}".format(epoch_acc))

        test(model, test_loader, criterion, use_cuda)

    return model

def net(use_cuda):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.inception_v3(pretrained=True)

    # Freeze params in feature model.
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 256),
                             nn.ReLU(inplace=True),
                             nn.Linear(256, 133)
    )
    model.aux_logits = False

    fc_params = model.fc.parameters()

    for param in fc_params:
        param.requires_grad = True

    if use_cuda:
        model = model.cuda()

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    # Standard Normalization.
    standard_normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.Resize(299),
                                          transforms.CenterCrop(299),
                                          transforms.ToTensor(),
                                          standard_normalization])

    test_transform = transforms.Compose([transforms.Resize(299),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      standard_normalization])


    train_data = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)

    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
     # Check if CUDA is available.
    use_cuda = torch.cuda.is_available()
    model=net(use_cuda)


    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batchsize)

    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.5)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Start training")
    model=train(model, train_loader, test_loader, criterion, optimizer, use_cuda)

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing model")
    test(model, test_loader, criterion, use_cuda)

    '''
    TODO: Save the trained model
    '''
    logger.info("Saving model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )

    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()

    main(args)
