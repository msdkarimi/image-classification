import argparse
import os
import logging
import torch

if not os.path.exists(os.path.join(os.getcwd(), "outputDir")):
    os.mkdir(os.path.join(os.getcwd(), "outputDir"))
    os.mkdir(os.path.join(os.path.join(os.getcwd(), "outputDir"), "logs"))
    os.mkdir(os.path.join(os.path.join(os.getcwd(), "outputDir"), "weights"))

parser = argparse.ArgumentParser()
parser.add_argument("phase", help="whether train/eval/test", choices=["train", "validation", "test", "plot"])
parser.add_argument('--batchSize', type=int, default=256)
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ValFreq', type=int, default=10)
parser.add_argument('--stepSize', type=int, default=20, help='scheduler step size to reduce learning rate')
parser.add_argument('--workers', type=int, default=4, help='number of tasks that can be run in parallel.')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Hyperparameter for SGD.')
parser.add_argument('--wDecay', type=float, default=5e-5, help='value of weigh decay.')
parser.add_argument('--gamma', type=float, default=1e-1, help='Hyperparameter for scheduler.')
parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
parser.add_argument("--output", help="location of outputDir, logs or weights", default=os.path.join(os.getcwd(),
                                                                                                  'outputDir'))
parser.add_argument("--datasetDir", help="location dataset", default=os.path.join(os.getcwd(), "dataset"))

args = parser.parse_args()

if args.output != os.path.join(os.getcwd(), "outputDir"):
    if not os.path.exists(os.path.join(os.getcwd(), args.output)):
        os.mkdir(os.path.join(os.getcwd(), args.output))
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), args.output), "logs")):
            os.mkdir(os.path.join(os.path.join(os.getcwd(), args.output), "logs"))
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), args.output), "weights")):
            os.mkdir(os.path.join(os.path.join(os.getcwd(), args.output), "weights"))
        args.output = os.path.join(os.getcwd(), args.output)

    else:
        args.output = os.path.join(os.getcwd(), args.output)
#
if not args.cpu:
    assert torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'


logging.basicConfig(filename=f'{args.output}/logs/logs.txt',filemode="a", level=logging.INFO, format='> %(name)s | %(asctime)s | %(message)s')
logger = logging.getLogger(args.phase)

# logger.info("------------------------------START-------------------------")
#
# logger.info("-------------------------------END--------------------------")



