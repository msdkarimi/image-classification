import torch
from tqdm import tqdm
from custom_dataset import CustomDataset, split_date
from torchvision import transforms
from torch.utils.data import DataLoader
from experiment import Experiment, plots
from arguments import logger, args
import numpy as np
import os
import sys


def main():
    transformer = transforms.Compose([transforms.Resize(256),  # Resizes short size of the PIL image to 256
                                            transforms.CenterCrop(224),  # Crops a central square patch of the image
                                            # 224 because torchvision's AlexNet needs a 224x224 input!
                                            # Remember this when applying different transformations, otherwise you get an error
                                            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            # Normalizes tensor with mean and standard deviation
                                      ])
    train, val = split_date(address='train.txt')
    train = CustomDataset(imageAddresses=train, root='dataset', transformer=transformer)
    validation = CustomDataset(imageAddresses=val, root='dataset', transformer=transformer)
    test = CustomDataset(imageAddresses='test.txt', root='dataset', transformer=transformer)

    trainDataLoader = DataLoader(dataset=train, batch_size=args.batchSize, shuffle=True, num_workers=args.workers, drop_last=True)
    validationDataLoader = DataLoader(dataset=validation, batch_size=args.batchSize, shuffle=False, num_workers=args.workers)
    testDataLoader = DataLoader(dataset=test, batch_size=args.batchSize, shuffle=False, num_workers=args.workers)

    experiments = Experiment(num_classes=len(test.all_categories))
    device = 'cpu' if args.cpu else 'cuda:0'

    if args.phase == 'train':

        if os.path.exists(os.path.join(args.output, 'weights', 'best_checkpoint.pth')):
            best_loss, epoch = experiments.load_checkpoint(os.path.join(args.output, 'weights', 'best_checkpoint.pth'))
            logger.info(f'-----------------RESUME---TRAIN----FROM----EPOCH----#{epoch}')
            epochs = args.epochs - epoch

        else:
            logger.info("-----------------START---TRAIN---WITH---BATCH-NORM")
            best_loss = 1e10
            epoch = 0
            epochs = args.epochs

        for e in range(epochs):
            loss = 0
            counter = 0

            # for a_batch in trainDataLoader:
            #     loss += experiments.train(a_batch)
            #     counter += a_batch[1].shape[0]
            with tqdm(trainDataLoader, unit='batch') as pbar:

                if (e + epoch) % args.ValFreq == 0:
                    accuracy_validation = experiments.validation(data=validationDataLoader)
                    accuracy_train = experiments.validation(data=trainDataLoader)
                    logger.info(f'-------------VALIDATION-ON-TRAIN-SET---{accuracy_train}')
                    logger.info(f'-------------VALIDATION-ON-VALIDATION-SET---{accuracy_validation}')

                pbar.set_description(f'train phase - epoch {e + epoch}')

                for a_batch in pbar:
                    loss += experiments.train(a_batch)
                    counter += a_batch[1].shape[0]
                    # pbar.set_description(f'train phase - epoch {e + epoch}')

                loss /= len(trainDataLoader)
                logger.info(f'---------------------LOSS-OF-EPOCH-FOR-TRAIN-#{loss}#')
                _, loss_val_per_epoch = experiments.validation(validationDataLoader)
                logger.info(f'---------------------LOSS-OF-EPOCH-FOR-VALIDATION-#{loss_val_per_epoch}#')

                if loss < best_loss:
                    best_loss = loss
                    experiments.save_checkpoint(loss, e+epoch+1, os.path.join(args.output, 'weights', 'best_checkpoint.pth'))

                experiments.scheduler.step()
                # print(f'epoch->>{e}')

        logger.info("---------END-OF-TRAIN")

    elif args.phase == 'validation':
        accuracy, loss_val = experiments.load_checkpoint(os.path.join(args.output, 'weights', 'best_checkpoint.pth'),
                                           validation=validationDataLoader)
        # if os.path.exists(os.path.join(args.output, 'weights', 'best_checkpoint.pth')):
        #     logger.info(f'-------------VALIDATION----ON----VALIDATION-SET')
        #     _, _ = experiments.load_checkpoint(os.path.join(args.output, 'weights', 'best_checkpoint.pth'))
        #     accuracy, loss_val = experiments.validation(validationDataLoader)
        #     logger.info(f'-----------------ACCURACY-OF-VALIDATION-{accuracy}---LOSS-OF-VALIDATION-#{loss_val}#')
        # else:
        #     logger.exception("-----------------Exception,To Start Validation Having a Checkpoint is REQUIRED")

    else:
        accuracy, loss_eval = experiments.load_checkpoint(os.path.join(args.output, 'weights', 'best_checkpoint.pth'),
                                           test=testDataLoader)
        # if os.path.exists(os.path.join(args.output, 'weights', 'best_checkpoint.pth')):
        #     logger.info(f'-----------------Evaluation----ON----Test-SET')
        #     _, _ = experiments.load_checkpoint(os.path.join(args.output, 'weights', 'best_checkpoint.pth'))
        #     accuracy, loss_eval = experiments.validation(testDataLoader)
        #     logger.info(f'-------------ACCURACY--OF-EVALUATION-{accuracy}---LOSS-OF-EVALUATION-#{loss_eval}#')
        # else:
        #     logger.exception("-----------------Exception,To Start Evaluation Having a Checkpoint is REQUIRED")

if __name__ == '__main__':

    if args.phase == "train" or args.phase == "validation" or args.phase == "test" :
        main()
    else:
        plots(os.path.join(args.output, 'logs','logs.txt'))
    sys.exit(0)
