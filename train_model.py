import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile
import smdebug.pytorch as smd


ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, use_cuda, hook):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)


    test_loss=0
    correct=0
    total = 0.0

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        # Move to GPU.
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        output=model(inputs)
        loss=criterion(output, labels)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
        total += inputs.size(0)

    logger.info('Test Loss: {:.6f}\n'.format(test_loss))

    logger.info('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

def train(model, train_loader, test_loader, criterion, optimizer, use_cuda, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    hook.set_mode(smd.modes.TRAIN)

    epochs = 25
    train_loss = 0

    for epoch in range(1, epochs+1):

        # Model in training mode, dropout is on
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move to GPU.
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Training Loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        test(model, test_loader, criterion, use_cuda, hook)

    return model

def net(use_cuda):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # Use inception v3 pretrained model.
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

    # Register SMDebug hook.
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batchsize)

    # Create your loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.5)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Start training")
    model=train(model, train_loader, test_loader, criterion, optimizer, use_cuda, hook)

    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing model")
    test(model, test_loader, criterion, use_cuda, hook)

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
