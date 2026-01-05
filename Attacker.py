import foolbox
import torch
from tqdm import tqdm
import numpy as np


class Attacker:
    def __init__(self, pytorch_model, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.f_model = foolbox.models.PyTorchModel(pytorch_model, bounds=(0, 1), device=self.device)
        self.attack = foolbox.attacks.L2DeepFoolAttack()

    def cal_all_distance(self, dataloader):
        distances = []
        for image, _ in tqdm(dataloader):
            distance, _ = self.cal_distance(image)
            distances.extend(distance.tolist())
        return distances

    def cal_distance(self, batch_image):
        batch_image = batch_image.to(self.device)
        dnn_label = self.f_model(batch_image).argmax(dim=1)
        attack_results = self.attack(self.f_model, batch_image, dnn_label, epsilons=np.inf)
        distance = torch.norm((attack_results[0] - batch_image).view(batch_image.size(0), -1), dim=1)
        distance = distance.detach().cpu()
        return distance, dnn_label


if __name__ == '__main__':
    from ModelManager import ModelManager
    from DatasetManager import DatasetManager

    dataset = DatasetManager.create_dataset('CIFAR10', bug_id=0, batch_size=32)
    modelmanager = ModelManager()
    model = modelmanager.create_model(model_name='ResNet', dataset=dataset, device=1, mode='load', bug_id=0)
    attacker = Attacker(model.model_structure, device=1)
    image_distances = attacker.cal_all_distance(dataset.train_loader)
    dataset = DatasetManager.create_dataset('CIFAR10', bug_id=11, batch_size=1)
    for image_in, ground_label in dataset.train_loader:
        image_distance, prediction = attacker.cal_distance(image_in)
        print(image_distance)
        break
