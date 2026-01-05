import random
import numpy as np
import torch
import warnings
from ModelManager import ModelManager
from DatasetManager import DatasetManager
# filter out warnings
warnings.filterwarnings("ignore", category=UserWarning)
# clear GPU cache
torch.cuda.empty_cache()

from matplotlib import pyplot as plt

def plot_fig(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    device = 0
    mode = 'evaluate'
    for dataset_name in ['ATS', 'CTSD', 'GTSRB', 'TSCD', 'TTS']:
        for model_name in ['resnet', 'alexnet', 'densenet', 'squeezenet', 'vgg16']:
            for bug_id in range(0, 11):
                print('='*100)
                print(f"Model: {model_name}, Dataset: {dataset_name}, BugID: {bug_id}")
                dataset = DatasetManager.create_dataset(dataset_name=dataset_name, bug_id=bug_id, batch_size=256)
                model = ModelManager.create_model(model_name=model_name, dataset=dataset, mode=mode, bug_id=bug_id, device=device)
