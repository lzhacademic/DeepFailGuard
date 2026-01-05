from Attacker import Attacker
from ModelManager import ModelManager
from DatasetManager import DatasetManager
import pickle
from utils import check_dir, batch_infer
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def cal_confidence_for_batch(model, dataloader, device):
    attacker = Attacker(model.model_structure, device=device)
    distances = []
    for image, _ in tqdm(dataloader):
        distance, _ = attacker.cal_distance(image)
        distances.extend(distance.tolist())
    model.model_structure.to(device)
    dnn_labels = batch_infer(model.model_structure, dataloader, device=device)
    return distances, dnn_labels


def check_existence(model_name, dataset_name, bug_id):
    root_save_path = f"input_information/test_information_cached"
    if os.path.exists(os.path.join(root_save_path, f"{dataset_name}_{model_name}_bug{bug_id}_distances.npy")) and \
            os.path.exists(os.path.join(root_save_path, f"{dataset_name}_{model_name}_bug{bug_id}_dnn_labels.npy")):
        return True


def save_distances(distances, dnn_labels, model_name, dataset_name, bug_id):
    root_save_path = f"input_information/test_information_cached"
    check_dir(root_save_path)
    np.save(os.path.join(root_save_path, f"{dataset_name}_{model_name}_bug{bug_id}_distances.npy"), np.array(distances))
    np.save(os.path.join(root_save_path, f"{dataset_name}_{model_name}_bug{bug_id}_dnn_labels.npy"), np.array(dnn_labels))


batch_size = 8
device = 0
for dataset_name in ['CIFAR10']:
    for model_name in ['resnet', 'alexnet', 'densenet', 'squeezenet', 'vgg16']:
        for bug_id in range(13):
            print('='*100)
            print(f"Model: {model_name}, Dataset: {dataset_name}, BugID: {bug_id}, device: {device}")
            if check_existence(model_name, dataset_name, bug_id):
                continue
            dataset = DatasetManager.create_dataset(dataset_name=dataset_name, bug_id=bug_id, batch_size=batch_size)
            model = ModelManager.create_model(model_name=model_name, dataset=dataset, mode='load', bug_id=bug_id, device=device)
            # distances, dnn_labels = cal_confidence_for_batch(model, dataset.test_loader, device=device)
            dnn_labels = cal_confidence_for_batch(model, dataset.test_loader, device=device)
            save_distances(dnn_labels, model_name, dataset_name, bug_id)

for dataset_name in ['ATS', 'CTSD', 'GTSRB', 'TSCD', 'TTS', 'COVIDX9A']:
    for model_name in ['resnet', 'alexnet', 'densenet', 'squeezenet', 'vgg16']:
        for bug_id in range(11):
            print('='*100)
            print(f"Model: {model_name}, Dataset: {dataset_name}, BugID: {bug_id}, device: {device}")
            if check_existence(model_name, dataset_name, bug_id):
                continue
            dataset = DatasetManager.create_dataset(dataset_name=dataset_name, bug_id=bug_id, batch_size=batch_size)
            model = ModelManager.create_model(model_name=model_name, dataset=dataset, mode='load', bug_id=bug_id, device=device)
            distances, dnn_labels = cal_confidence_for_batch(model, dataset.test_loader, device=device)
            save_distances(distances, dnn_labels, model_name, dataset_name, bug_id)