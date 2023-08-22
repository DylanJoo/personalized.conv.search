import sys
import multiprocessing
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausualLM,
    HfArgumentParser,
    GenerationConfig
)
from datasets import load_dataset

# customized modules
from data import DataCollatorForStarter
from models import GTREncoder
from trainers import TrainerForStart
from arguments import ModelArgs, DataArgs, TrainArgs

import os

datacollator = DataCollatorForStarter(
        pass
)
