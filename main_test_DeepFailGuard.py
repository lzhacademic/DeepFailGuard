import random
import numpy as np
import torch
import warnings
from ModelManager import ModelManager
from DatasetManager import DatasetManager
from DeepFailGuard import DeepFailGuardModule, DeepFailGuard
from utils import val_acc_map, test_acc_map, nvp_results, check_dir, cal_upper_limit, cal_available_model

# set seed for reproducibility
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# filter out warnings
warnings.filterwarnings("ignore", category=UserWarning)
# clear GPU cache
torch.cuda.empty_cache()


def validate_dfg(dataset_name, model_names, bug_ids, batch_size, device):
    datasets = []
    dfg_modules = []
    test_accuracy = []
    for model_name, bug_id in zip(model_names, bug_ids):
        if bug_id in [7, 10, 11, 12]:
            dataset = DatasetManager.create_dataset(dataset_name, bug_id=bug_id, batch_size=batch_size)
        else:
            dataset = DatasetManager.create_dataset(dataset_name, batch_size=batch_size)
        datasets.append(dataset)
        model = ModelManager.create_model(model_name, dataset, mode='load', bug_id=bug_id, device=device)
        key = f"{model_name}_{dataset_name}_bug{bug_id}" if bug_id in range(1, 10) else f"{model_name}_{dataset_name}"
        correct_kde_path = f"input_information/kde_results/{key}_correct.pkl"
        wrong_kde_path = f"input_information/kde_results/{key}_wrong.pkl"
        val_acc = val_acc_map(model_name, dataset_name, bug_id)
        test_accuracy.append(test_acc_map(model_name, dataset_name, bug_id))
        module = DeepFailGuardModule(model=model, correct_kde_path=correct_kde_path, wrong_kde_path=wrong_kde_path, val_acc=val_acc, device=device)
        dfg_modules.append(module)

    dfg = DeepFailGuard(dfg_modules)
    # dfg_labels, all_model_labels = dfg.batch_infer([dataset.test_loader for dataset in datasets])
    dfg_labels, all_model_labels = dfg.batch_infer_cached([dataset for dataset in datasets])
    baseline_labels = nvp_results(all_model_labels)
    ground_truth = datasets[0].test_loader.dataset.targets
    upper_limit = cal_upper_limit(all_model_labels, ground_truth)
    accuracy = np.mean(np.array(dfg_labels) == np.array(ground_truth))
    baseline_accuracy = np.mean(np.array(baseline_labels) == np.array(ground_truth))

    print_str = f"{'=' * 50}\n" \
                f"Models: {model_names}\n" \
                f"Dataset: {dataset_name}\n" \
                f"bug_ids: {bug_ids}\n" \
                f"model_test_acc: {test_accuracy}\n" \
                f"dfg_accuracy: {accuracy:.4f}, baseline_accuracy: {baseline_accuracy:.4f}, upper_limit: {upper_limit:.4f}\n" \
                f"absolute improvement rate: {(accuracy-baseline_accuracy):.4f}, relative improvement rate: {((accuracy-baseline_accuracy)/(upper_limit-baseline_accuracy)):.4f}\n" \
                f"{'='*50}\n"
    print(print_str)
    check_dir('./dfg_results')
    with open(f'dfg_results/{dataset_name}_{len(model_names)}VersionDFT.txt', 'a+') as f:
        f.write(print_str)



def select_random_models(dataset_name, bug_ids):
    selected_models = []
    used_models = set()

    for i, bug_id in enumerate(bug_ids):
        available_models = cal_available_model(dataset_name, bug_id)
        available_models = [model for model in available_models if model not in used_models]

        if not available_models:
            print(f"No enough available models for bug_id {bug_id}")
            return None

        selected_model = random.choice(available_models)
        selected_models.append(selected_model)
        used_models.add(selected_model)

    return selected_models


if __name__ == '__main__':
    batch_size = 32
    device = 1
    print("*-"*100)
    dataset_name = 'CIFAR10'
    bug_id = [10, 0, 0]
    model_name = ['vgg16', 'resnet', 'densenet',]
    validate_dfg(dataset_name, model_name, bug_id, batch_size, device)