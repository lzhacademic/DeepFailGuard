from Attacker import Attacker
from scipy.stats import gaussian_kde
import pickle
from utils import check_dir, batch_infer
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class DeepFailGuardTrainer:
    @classmethod
    def train(cls, pytorch_model, val_dataloader, correct_kde_path, wrong_kde_path, device=0):
        attacker = Attacker(pytorch_model, device=device)
        distances = attacker.cal_all_distance(val_dataloader)
        predictions = batch_infer(pytorch_model, val_dataloader, device=device)
        labels = val_dataloader.dataset.targets
        correct_distances = [d for d, p, l in zip(distances, predictions, labels) if p == l]
        wrong_distances = [d for d, p, l in zip(distances, predictions, labels) if p != l]
        if len(set(correct_distances)) == 1:
            correct_distances.append(correct_distances[0] + 0.0001)
        if len(set(wrong_distances)) == 1:
            wrong_distances.append(wrong_distances[0] + 0.0001)
        correct_kde = gaussian_kde(correct_distances)
        wrong_kde = gaussian_kde(wrong_distances)
        cls.save_kde(correct_kde, correct_kde_path)
        cls.save_kde(wrong_kde, wrong_kde_path)
        return correct_kde, wrong_kde

    @staticmethod
    def save_kde(kde, path):
        check_dir(path)
        with open(path, 'wb') as f:
            pickle.dump(kde, f)


class DeepFailGuardModule:
    def __init__(self, model, correct_kde_path, wrong_kde_path, val_acc, device=0):
        self.model = model
        self.attacker = Attacker(self.model.model_structure, device=device)
        self.correct_kde, self.wrong_kde = self.load_kde(correct_kde_path), self.load_kde(wrong_kde_path)
        self.val_acc = val_acc

    @staticmethod
    def load_kde(path):
        if not (os.path.exists(path)):
            return None
        with open(path, 'rb') as f:
            kde = pickle.load(f)
        return kde

    @staticmethod
    def plot_kde(kde, save_path=None):
        x = np.linspace(0, 3, 100)
        y = kde(x)
        plt.plot(x, y)
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.legend()
        if save_path:
            check_dir(save_path)
            plt.savefig(save_path)
        plt.show()

    def cal_confidence(self, image):
        distance, dnn_label = self.attacker.cal_distance(image)
        correct_prob = self.correct_kde(distance) * self.val_acc
        wrong_prob = self.wrong_kde(distance) * (1 - self.val_acc)
        return correct_prob / (correct_prob + wrong_prob), dnn_label

    def cal_confidence_for_batch(self, dataloader):
        distances, dnn_labels = [], []
        for image, _ in tqdm(dataloader):
            distance, dnn_label = self.attacker.cal_distance(image)
            distances.extend(distance.tolist())
            dnn_labels.extend(dnn_label.tolist())
        confidences = []
        for i in range(len(distances)):
            correct_prob = self.correct_kde(distances[i]) * self.val_acc
            wrong_prob = self.wrong_kde(distances[i]) * (1 - self.val_acc)
            # check
            if correct_prob + wrong_prob == 0:
                if correct_prob == 0:
                    correct_prob = np.log(1+distances[i]) * self.val_acc
                if wrong_prob == 0:
                    wrong_prob = 1 - correct_prob
            confidences.append(correct_prob / (correct_prob + wrong_prob))
        return confidences, dnn_labels

    def cal_confidence_for_batch_cached(self, dataset):
        root_save_path = f"input_information/test_information_cached"
        model_name = self.model.model_name
        dataset_name = dataset.dataset_name
        bug_id = dataset.bug_id
        distances = np.load(os.path.join(root_save_path, f"{dataset_name}_{model_name}_bug{bug_id}_distances.npy"))
        dnn_labels = np.load(os.path.join(root_save_path, f"{dataset_name}_{model_name}_bug{bug_id}_dnn_labels.npy"))

        confidences = []
        for i in range(len(distances)):
            correct_prob = self.correct_kde(distances[i]) * self.val_acc
            wrong_prob = self.wrong_kde(distances[i]) * (1 - self.val_acc)
            # check
            if correct_prob + wrong_prob == 0:
                if correct_prob == 0:
                    correct_prob = np.log(1+distances[i]) * self.val_acc
                if wrong_prob == 0:
                    wrong_prob = 1 - correct_prob
            confidences.append(correct_prob / (correct_prob + wrong_prob))
        return confidences, dnn_labels


class DeepFailGuard:
    def __init__(self, deep_fail_guard_modules):
        self.modules = deep_fail_guard_modules

    def infer(self, image):
        correct_prob_max, output_label = -np.inf, None
        for m in self.modules:
            correct_prob, prediction = m.cal_confidence(image)
            if correct_prob > correct_prob_max:
                correct_prob_max = correct_prob
                output_label = prediction
        return output_label

    def batch_infer(self, dataloaders):
        all_module_confidences, all_module_dnn_labels = [], []
        for m, dataloader in zip(self.modules, dataloaders):
            confidences, dnn_labels = m.cal_confidence_for_batch(dataloader)
            all_module_confidences.append(confidences)
            all_module_dnn_labels.append(dnn_labels)
        outputs = []
        for i in range(len(all_module_dnn_labels[0])):
            max_confidence, output_label = -np.inf, None
            for j in range(len(self.modules)):
                if all_module_confidences[j][i] > max_confidence:
                    max_confidence = all_module_confidences[j][i]
                    output_label = all_module_dnn_labels[j][i]
            outputs.append(output_label)
        return outputs, all_module_dnn_labels

    def batch_infer_cached(self, datasets):
        all_module_confidences, all_module_dnn_labels = [], []
        for m, dataset in zip(self.modules, datasets):
            confidences, dnn_labels = m.cal_confidence_for_batch_cached(dataset)
            all_module_confidences.append(confidences)
            all_module_dnn_labels.append(dnn_labels)
        outputs = []
        for i in range(len(all_module_dnn_labels[0])):
            max_confidence, output_label = -np.inf, None
            for j in range(len(self.modules)):
                if all_module_confidences[j][i] > max_confidence:
                    max_confidence = all_module_confidences[j][i]
                    output_label = all_module_dnn_labels[j][i]
            outputs.append(output_label)
        return outputs, all_module_dnn_labels
