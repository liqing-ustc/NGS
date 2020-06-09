import os
import argparse
from train import train_model

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--mode', default='BS', type=str, help='choose mode. BS or RL or MAPO' )
parser.add_argument('--nstep', default=5, type=int, help='number of steps of backsearching')
parser.add_argument('--pretrain', default=None, type=str, help='pretrained symbol net')
# Dataloader
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--batch_size', default=64, type=int)
# seed
parser.add_argument('--random_seed', default=123, type=int, help="numpy random seed")
parser.add_argument('--manual_seed', default=17, type=int, help="torch manual seed")
# Run
parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
parser.add_argument('--decay', default=0.99, type=float, help="reward decay")
parser.add_argument('--num_epochs', default=5, type=int, help="number of epochs")
parser.add_argument('--n_epochs_per_eval', default=1, type=int, help="test every n epochs")
parser.add_argument('--output_dir', default='output', type=str, help="output directory")

opt = parser.parse_args()
print(opt)
train_model(opt)