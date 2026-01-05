import pickle
from scipy.stats import mode
import random
import numpy as np
import torch
import warnings
from utils import val_acc_map, test_acc_map, nvp_results, check_dir, cal_upper_limit, cal_available_model
import os

# set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# filter out warnings
warnings.filterwarnings("ignore", category=UserWarning)
# clear GPU cache
torch.cuda.empty_cache()

class DFGModule:
    def __init__(self, model_name, dataset_name, bug_id, info=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.bug_id = bug_id
        self.distance, self.dnn_label, self.correct_kde, self.wrong_kde, self.val_acc, self.test_acc = self.load_information()
        if not info:
            self.confidence = self.estimate_confidence()
        elif info == 'distance_v1':
            self.confidence = self.estimate_confidence_by_distance_v1()
        elif info == 'distance_v2':
            self.confidence = self.estimate_confidence_by_distance_v2()
        elif info == 'distance_v3':
            self.confidence = self.estimate_confidence_by_distance_v3()
        elif info == 'distance_v4':
            self.confidence = self.estimate_confidence_by_distance_v4()
        elif info == 'val_acc':
            self.confidence = self.estimate_confidence_by_val_acc()

    def load_information(self):
        distance_file = f"input_information/test_information_cached/{self.dataset_name}_{self.model_name}_bug{self.bug_id}_distances.npy"
        label_file = f"input_information/test_information_cached/{self.dataset_name}_{self.model_name}_bug{self.bug_id}_dnn_labels.npy"
        key = f"{self.model_name}_{self.dataset_name}_bug{self.bug_id}" if self.bug_id in range(1, 10) else f"{self.model_name}_{self.dataset_name}"
        correct_kde_path = f"input_information/kde_results/{key}_correct.pkl"
        wrong_kde_path = f"input_information/kde_results/{key}_wrong.pkl"
        distance = np.load(distance_file)
        dnn_label = np.load(label_file)
        with open(correct_kde_path, 'rb') as f:
            correct_kde = pickle.load(f)
        with open(wrong_kde_path, 'rb') as f:
            wrong_kde = pickle.load(f)
        val_acc = val_acc_map(self.model_name, self.dataset_name, self.bug_id)
        test_acc = test_acc_map(self.model_name, self.dataset_name, self.bug_id)
        return distance, dnn_label, correct_kde, wrong_kde, val_acc, test_acc

    def estimate_confidence(self):
        confidence = []
        for i in range(len(self.distance)):
            correct_prob = self.correct_kde(self.distance[i]) * self.val_acc
            wrong_prob = self.wrong_kde(self.distance[i]) * (1 - self.val_acc)
            if correct_prob + wrong_prob == 0:
                if correct_prob == 0:
                    correct_prob = np.log(1 + self.distance[i]) * self.val_acc
                if wrong_prob == 0:
                    wrong_prob = 1 - correct_prob
            confidence.append(correct_prob / (correct_prob + wrong_prob))
        return confidence

    def estimate_confidence_by_distance_v1(self):
        confidence = []
        for i in range(len(self.distance)):
            correct_prob = self.correct_kde(self.distance[i])
            wrong_prob = self.wrong_kde(self.distance[i])
            confidence.append(correct_prob - wrong_prob)
        return confidence

    def estimate_confidence_by_distance_v2(self):
        confidence = []
        for i in range(len(self.distance)):
            correct_prob = self.correct_kde(self.distance[i])
            wrong_prob = self.wrong_kde(self.distance[i])
            confidence.append(correct_prob / (correct_prob + wrong_prob))
        return confidence

    def estimate_confidence_by_distance_v3(self):
        confidence = []
        for i in range(len(self.distance)):
            correct_prob = self.correct_kde(self.distance[i])
            confidence.append(correct_prob)
        return confidence

    def estimate_confidence_by_distance_v4(self):
        confidence = []
        for i in range(len(self.distance)):
            wrong_prob = self.wrong_kde(self.distance[i])
            confidence.append(1 - wrong_prob)
        return confidence

    def estimate_confidence_by_val_acc(self):
        confidence = []
        for i in range(len(self.distance)):
            correct_prob = self.val_acc
            confidence.append(correct_prob)
        return confidence


class FaultTolerancer:
    @staticmethod
    def DFG_predict(modules):
        outputs = []
        for i in range(len(modules[0].dnn_label)):
            model_correct_probs = [module.confidence[i] for module in modules]
            argmax_index = model_correct_probs.index(max(model_correct_probs))
            outputs.append(modules[argmax_index].dnn_label[i])
        return outputs

    @staticmethod
    def Voting_predict(modules):
        all_model_labels = [module.dnn_label for module in modules]
        all_model_labels = np.array(all_model_labels)
        voting_labels = mode(all_model_labels, axis=0)[0]
        return voting_labels


def result_analysis(model_names, dataset_name, bug_ids, info=None):
    if bug_ids[0] in (11, 12):
        ground_truth = np.load(f"input_information/test_information_cached/{dataset_name}_bug{bug_ids[0]}_ground_truth.npy")
    else:
        ground_truth = np.load(f"input_information/test_information_cached/{dataset_name}_ground_truth.npy")

    modules = [DFGModule(model_name, dataset_name, bug_id, info) for model_name, bug_id in zip(model_names, bug_ids)]
    dfg_labels = FaultTolerancer.DFG_predict(modules)
    baseline_labels = FaultTolerancer.Voting_predict(modules)
    all_model_labels = [module.dnn_label for module in modules]

    upper_limit = cal_upper_limit(all_model_labels, ground_truth)

    dfg_acc = np.mean(np.array(dfg_labels) == np.array(ground_truth))
    baseline_acc = np.mean(np.array(baseline_labels) == np.array(ground_truth))

    print_str = f"{'=' * 50}\n" \
                f"Dataset: {dataset_name}\n" \
                f"Models: {model_names}\n" \
                f"bug_ids: {bug_ids}\n" \
                f"model_test_acc: {[model.test_acc for model in modules]}\n" \
                f"dfg_accuracy: {dfg_acc:.4f}, baseline_accuracy: {baseline_acc:.4f}, upper_limit: {upper_limit:.4f}\n" \
                f"absolute improvement rate: {(dfg_acc - baseline_acc):.4f}, relative improvement rate: {((dfg_acc - baseline_acc) / (upper_limit - baseline_acc)):.4f}\n"

    if len(model_names) == 3:
        absolute_wrong_1 = (all_model_labels[0]!=ground_truth)&(all_model_labels[1]==ground_truth)&(all_model_labels[2]==ground_truth)
        absolute_wrong_2 = (all_model_labels[1]!=ground_truth)&(all_model_labels[0]==ground_truth)&(all_model_labels[2]==ground_truth)
        absolute_wrong_3 = (all_model_labels[2]!=ground_truth)&(all_model_labels[0]==ground_truth)&(all_model_labels[1]==ground_truth)
        absolute_wrong_12 = (all_model_labels[0]!=ground_truth)&(all_model_labels[1]!=ground_truth)&(all_model_labels[2]==ground_truth)
        absolute_wrong_13 = (all_model_labels[0]!=ground_truth)&(all_model_labels[2]!=ground_truth)&(all_model_labels[1]==ground_truth)
        absolute_wrong_23 = (all_model_labels[1]!=ground_truth)&(all_model_labels[2]!=ground_truth)&(all_model_labels[0]==ground_truth)
        all_correct = (all_model_labels[0]==ground_truth)&(all_model_labels[1]==ground_truth)&(all_model_labels[2]==ground_truth)
        all_wrong = (all_model_labels[0]!=ground_truth)&(all_model_labels[1]!=ground_truth)&(all_model_labels[2]!=ground_truth)
        e1 = np.sum(absolute_wrong_1) + np.sum(absolute_wrong_2) + np.sum(absolute_wrong_3)
        e2 = np.sum(absolute_wrong_12) + np.sum(absolute_wrong_13) + np.sum(absolute_wrong_23)
        e3 = np.sum(all_wrong)
        lower_limit = np.sum(all_correct)/len(all_model_labels[0])

        e2_baseline_correct = (absolute_wrong_12 | absolute_wrong_23 | absolute_wrong_13) & (np.array(baseline_labels) == np.array(ground_truth))
        e2_DFG_correct = (absolute_wrong_12 | absolute_wrong_23 | absolute_wrong_13) & (dfg_labels == ground_truth)
        e2_DFG_revise = e2_DFG_correct & (np.array(baseline_labels) != np.array(ground_truth))

        e1_baseline_correct = (absolute_wrong_1 | absolute_wrong_2 | absolute_wrong_3) & (np.array(baseline_labels) == np.array(ground_truth))
        e1_DFG_correct = (absolute_wrong_1 | absolute_wrong_2 | absolute_wrong_3) & (dfg_labels == ground_truth)
        e1_DFG_revise = e1_DFG_correct & (np.array(baseline_labels) != np.array(ground_truth))

        print_str += f"absolute_wrong_1: {np.sum(absolute_wrong_1)}, absolute_wrong_2: {np.sum(absolute_wrong_2)}, absolute_wrong_3: {np.sum(absolute_wrong_3)}\n" \
                        f"absolute_wrong_12: {np.sum(absolute_wrong_12)}, absolute_wrong_13: {np.sum(absolute_wrong_13)}, absolute_wrong_23: {np.sum(absolute_wrong_23)}\n" \
                        f"all_correct: {np.sum(all_correct)}, all_wrong: {np.sum(all_wrong)}\n" \
                        f"e1: {e1}, e2: {e2}, e3: {e3}\n" \
                        f"e1_baseline_correct: {np.sum(e1_baseline_correct)}, e1_DFG_correct: {np.sum(e1_DFG_correct)}, e1_DFG_revise: {np.sum(e1_DFG_revise)}\n" \
                        f"e2_baseline_correct: {np.sum(e2_baseline_correct)}, e2_DFG_correct: {np.sum(e2_DFG_correct)}, e2_DFG_revise: {np.sum(e2_DFG_revise)}\n" \
                        f"lower_limit: {lower_limit}\n"
    print_str += f"{'=' * 50}\n"


    print(print_str)
    root_dir = f"dfg_results/"
    path = os.path.join(root_dir, info) if info else root_dir
    file_name = f"{len(model_names)}VersionsDFG_{dataset_name}.txt"
    file = os.path.join(path, file_name)

    check_dir(file)
    with open(file, 'a+') as f:
        f.write(print_str)
    print(f"Results have been written to {file}")


def select_random_models(dataset_name, bug_ids):
    selected_models = []
    used_models = set()

    for i, bug_id in enumerate(bug_ids):
        available_models = cal_available_model(dataset_name, bug_id)
        available_models = [model for model in available_models if model not in used_models]

        if not available_models:
            print(f"No enough available models for bug_id {bug_id}")
            break

        selected_model = random.choice(available_models)
        selected_models.append(selected_model)
        used_models.add(selected_model)

    return selected_models


def select_random_models_2to5(dataset_name, bug_ids):
    selected_models = select_random_models(dataset_name, bug_ids)
    if len(selected_models) == 5:
        return selected_models[:2], selected_models[:3], selected_models[:4], selected_models[:5]
    elif len(selected_models) == 4:
        return selected_models[:2], selected_models[:3], selected_models[:4], None
    elif len(selected_models) == 3:
        return selected_models[:2], selected_models[:3], None, None
    elif len(selected_models) == 2:
        return selected_models[:2], None, None, None
    else:
        return None, None, None, None



if __name__ == '__main__':
    # info = None: Complete DeepFailGuard
    # info = 'distance_v1': Distance Only
    # info = 'val_acc': Val_Acc Only
    info = None
    for dataset_name in ['CIFAR10', 'ATS', 'CTSD', 'GTSRB', 'TSCD', 'TTS', 'COVIDX9A']:
        if dataset_name == 'CIFAR10':
            for i in range(13):
                bug_ids = [i] * 5
                model_names = select_random_models_2to5(dataset_name, bug_ids)
                for m in model_names:
                    if m:
                        result_analysis(m, dataset_name, bug_ids[:len(m)], info)
        else:
            for i in range(11):
                bug_ids = [i] * 5
                model_names = select_random_models_2to5(dataset_name, bug_ids)
                for m in model_names:
                    if m:
                        result_analysis(m, dataset_name, bug_ids[:len(m)], info)
        for i in range(1, 11):
            bug_ids = [i, 0, 0, 0, 0]
            model_names = select_random_models_2to5(dataset_name, bug_ids)
            for m in model_names:
                if m:
                    result_analysis(m, dataset_name, bug_ids[:len(m)], info)

