"""
Main script to train and evaluate audio classification models.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import wandb
import numpy as np

from data_module import CustomDataModule
from models import CnnModel, CnnTransformerModel
from trainer import ModelTrainer


def set_seed(seed_value=19):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed_value)
    