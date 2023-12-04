import torch
from model import AlexNet
import torch.optim as optim
from arguments import args, logger
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess
import os
class Experiment:
    def __init__(self, num_classes=None):
        assert num_classes != None, "specify number of classes"
        self.model = AlexNet(num_classes=num_classes)

        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = optim.SGD(params=self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wDecay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.stepSize, gamma=args.gamma)
        self.criterion = nn.CrossEntropyLoss()
        self.device = 'cpu' if args.cpu else 'cuda:0'


    def train(self, a_batch):

        """
        :param a_batch: contains a batch of images and corresponding labels sgape ([[batch_size, images],[batch_size(which contains all labels)]])
        :return: returns loss of a single batch
        """
        self.model.train()
        net = self.model.to(self.device)

        images, labels = a_batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = net(images)
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss
        return loss.item()

    def validation(self, data, validation=None, test=None):

        if validation:
            logger.info(f'-------------VALIDATION----ON----VALIDATION-SET')
        elif test:
            logger.info(f'-----------------Evaluation----ON----Test-SET')

        self.model.eval()
        net = self.model.to()
        loss = 0
        accuracy = 0
        with tqdm(data, unit='batch') as pbar:
            pbar.set_description(f'validation frequency started')
            for a_batch in pbar:
                images, labels = a_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)

                loss += self.criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                true_positive = torch.sum(labels == predictions)
                batch_accuracy = true_positive.item()/a_batch[1].shape[0]
                accuracy+=batch_accuracy
        self.model.train()
        return f'%{(accuracy/len(data))*100}%', loss.item()/len(data)

    def load_checkpoint(self, checkpoint_path, validation=None, test=None):
        """
        :param validation: if in validation phase
        :param checkpoint_path:
        :return:
        """
        if os.path.exists(checkpoint_path):
            logger.info('-----------------Loading checkpoint!')
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            if validation is not None:
                accuracy, loss_val = self.validation(validation, validation=True)
                logger.info(f'-----------------ACCURACY-OF-VALIDATION-{accuracy}---LOSS-OF-VALIDATION-#{loss_val}#')
                return accuracy, loss_val

            elif test is not None:
                accuracy, loss_eval = self.validation(test, test=True)
                logger.info(f'-------------ACCURACY--OF-EVALUATION-{accuracy}---LOSS-OF-EVALUATION-#{loss_eval}#')
                return accuracy, loss_eval

            else:
                return checkpoint['loss'], checkpoint['epoch']

        else:
            logger.exception("-----------------Exception,To Start Validation Having a Checkpoint is REQUIRED")

        # return checkpoint['loss'], checkpoint['epoch']

    def save_checkpoint(self, loss, epoch, checkpoint_path):
        """
        :param loss:
        :param epoch:
        :param checkpoint_path:
        :return:
        """
        logger.info('-----------------Saving checkpoint!')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': loss,
            'epoch': epoch
        }, checkpoint_path)

def plots(logs):

    with open(logs, "r") as file:
        temp =  file.readlines()
    losses_train = [float(line.strip().split('#')[1]) for line in temp if 'LOSS-OF-EPOCH-FOR-TRAIN' in line]
    losses_validation = [float(line.strip().split('#')[1]) for line in temp if 'LOSS-OF-EPOCH-FOR-VALIDATION' in line]
    train_accuracy = [float(line.strip().split('%')[1]) for line in temp if 'VALIDATION-ON-TRAIN-SET' in line]
    validation_accuracy = [float(line.strip().split('%')[1]) for line in temp if 'VALIDATION-ON-VALIDATION-SET' in line]

    fig, (col1, col2) = plt.subplots(1, 2)
    fig.suptitle('Train and validation accuracy')
    col1.plot([i+1 for i in range(len(losses_train))], losses_train, color='red', label='train_loss')
    col1.plot([i+1 for i in range(len(losses_validation))], losses_validation, color='blue', label='validation_loss')
    col2.plot([i+1 for i in range(len(train_accuracy))], train_accuracy, color='red', label='train_acc')
    col2.plot([i+1 for i in range(len(validation_accuracy))], validation_accuracy, color='blue', label='validation_acc')

    tmp = list(range(len(train_accuracy)))[1:]
    tmp.append(len(train_accuracy))
    col2.set_xticks(tmp, np.arange(10, len(train_accuracy)*10+1, 10))

    col1.set_title('train loss')
    col1.legend()
    col2.set_title('train/validation accuracy')
    col2.legend()

    plt.tight_layout()

    plt.savefig('subplots.png')


def install_requirements(requirements_file):
    try:
        subprocess.run(["pip", "install", "-r", requirements_file], check=True)
        print("Successfully installed requirements!")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install requirements - {e}")

