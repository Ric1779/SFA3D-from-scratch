import time
import numpy as np
import sys
import random
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

from config.train_config import parse_train_configs

def main():
    configs = parse_train_configs()