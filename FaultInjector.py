import torch
import torch.nn as nn
import numpy as np
import imgaug.augmenters as iaa


class CustomCrossEntropyLoss(nn.Module):
    # bug6. wrong loss function
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, inputs, target):
        # Compute standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss(weight=self.weight,
                                      size_average=self.size_average,
                                      ignore_index=self.ignore_index,
                                      reduce=self.reduce,
                                      reduction=self.reduction)(inputs, target)

        mask = (target == 0).float()
        extra_loss = torch.sum(mask) * 0.1
        custom_loss = ce_loss + extra_loss

        return custom_loss


class FaultInjector:
    @staticmethod
    def inject_model_fault(model, bug_id):
        assert bug_id in [1, 2, 3], f"bugID {bug_id} not supported for model"
        # Inject fault to model according to bugID
        if bug_id == 1:
            # bug1. suboptimal network structure
            model.model_structure = model.change_model_structure()
        elif bug_id == 2:
            # bug2. wrong filter size for a conv layer
            for name, module in model.model_structure.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.stride == (1, 1):
                        module.kernel_size = (5, 5)
                        module.padding = 'same'
        elif bug_id == 3:
            # bug3. missing ReLU activation function
            for name, module in model.model_structure.named_modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = False

    @staticmethod
    def inject_hyperparameters_fault(trainer, bug_id):
        assert bug_id in [4, 5, 6], f"bugID {bug_id} not supported for hyperparameters"
        # Inject fault to hyperparameters according to bugID
        if bug_id == 4:
            # bug4. suboptimal learning rate
            trainer.optimizer.lr = 0.002
        elif bug_id == 5:
            # bug5. suboptimal number of epochs
            trainer.epochs = 10
        elif bug_id == 6:
            # bug6. wrong selection of loss function
            trainer.criteria = CustomCrossEntropyLoss()

    @staticmethod
    def inject_dataset_fault(dataset, bug_id):
        assert bug_id in [7, 8, 9, 10, 11, 12], f"bugID {bug_id} not supported for dataset"
        # Inject fault to dataset according to bugID
        if bug_id == 7:
            # bug7. wrong preprocessing
            dataset.x_train = dataset.x_train.astype('float32') / 255 / 2
            dataset.x_val = dataset.x_val.astype('float32') / 255 / 2
            dataset.x_test = dataset.x_test.astype('float32') / 255 / 2
        elif bug_id == 8:
            # bug8. wrong labels for training data
            np.random.shuffle(dataset.y_train[0:int(len(dataset.y_train)/10)])
        elif bug_id == 9:
            # bug9. not enough training data
            dataset.x_train = dataset.x_train[0:int(len(dataset.x_train)/2)]
            dataset.y_train = dataset.y_train[0:int(len(dataset.y_train)/2)]
        elif bug_id == 10:
            # bug10. add noise to test data
            if dataset.dataset_name == 'CIFAR10':
                seq = iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), seed=42)
                x_test = seq(images=dataset.x_test)
                dataset.x_test = x_test.clip(0, 255)
            elif dataset.dataset_name == 'COVIDX9A':
                seq_blur = iaa.GaussianBlur(sigma=(0.0, 1.5))
                x_test_blur = seq_blur(images=dataset.x_test)
                dataset.x_test = x_test_blur
            else:
                seq = iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 1), seed=42)
                x_test = seq(images=dataset.x_test)
                dataset.x_test = x_test.clip(0, 1)
        elif bug_id == 11:
            # bug11. CIFAR10.1_v4
            assert dataset.dataset_name == 'CIFAR10', f"bugID {bug_id} only supported for CIFAR10 dataset"
            dataset.x_test = np.load('mydataset/data/CIFAR10_1/cifar10.1_v4_data.npy')
            dataset.y_test = np.load('mydataset/data/CIFAR10_1/cifar10.1_v4_labels.npy').astype('int64')
        elif bug_id == 12:
            # bug12. CIFAR10.1_v6
            assert dataset.dataset_name == 'CIFAR10', f"bugID {bug_id} only supported for CIFAR10 dataset"
            dataset.x_test = np.load('mydataset/data/CIFAR10_1/cifar10.1_v6_data.npy')
            dataset.y_test = np.load('mydataset/data/CIFAR10_1/cifar10.1_v6_labels.npy').astype('int64')
