import random
import numpy as np
import torch
import warnings
from ModelManager import ModelManager
from DatasetManager import DatasetManager
from DeepFailGuard import DeepFailGuardTrainer
# set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# filter out warnings
warnings.filterwarnings("ignore", category=UserWarning)
# clear GPU cache
torch.cuda.empty_cache()
import os

if __name__ == '__main__':
    mode = 'load'
    device = 1
    batch_size = 8

    for dataset_name in ['CIFAR10', 'ATS', 'CTSD', 'GTSRB', 'TSCD', 'TTS', 'COVIDX9A']:
        for model_name in ['resnet', 'alexnet', 'densenet', 'squeezenet', 'vgg16']:
            for bug_id in range(10):
                print(f"Training {dataset_name}_{model_name}_bug{bug_id}")
                key = f"{model_name}_{dataset_name}_bug{bug_id}" if bug_id else f"{model_name}_{dataset_name}"
                correct_kde_path = f"input_information/kde_results/{key}_correct.pkl"
                wrong_kde_path = f"input_information/kde_results/{key}_wrong.pkl"
                if not os.path.exists(correct_kde_path) or not os.path.exists(wrong_kde_path):
                    dataset = DatasetManager.create_dataset(dataset_name=dataset_name, bug_id=bug_id, batch_size=batch_size)
                    model = ModelManager.create_model(model_name=model_name, dataset=dataset, mode=mode, bug_id=bug_id, device=device)
                    DeepFailGuardTrainer.train(pytorch_model=model.model_structure, val_dataloader=dataset.val_loader, correct_kde_path=correct_kde_path, wrong_kde_path=wrong_kde_path, device=device)
                    print(f'{dataset_name}_{model_name}_bug{bug_id} is trained')
                else:
                    print(f'{dataset_name}_{model_name}_bug{bug_id} is already trained')
                    pass
