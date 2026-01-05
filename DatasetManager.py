from mydataset import CIFAR10Data, ATSData, CTSDData, GTSRBData, TSCDData, TTSData, COVIDX9AData
from torch.utils.data import Dataset, DataLoader
from FaultInjector import FaultInjector


class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


def create_dataloader(x, y, transform, batch_size, shuffle):
    dataset = CustomDataset(x, y, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader


class DatasetManager:
    dataset_list = {}  # Manage the dataset dict

    @staticmethod
    def packaging_dataloader(dataset, batch_size):
        dataset.train_loader = create_dataloader(dataset.x_train, dataset.y_train, dataset.transform, batch_size=batch_size, shuffle=True)
        dataset.val_loader = create_dataloader(dataset.x_val, dataset.y_val, dataset.transform, batch_size=batch_size, shuffle=False)
        dataset.test_loader = create_dataloader(dataset.x_test, dataset.y_test, dataset.transform, batch_size=batch_size, shuffle=False)

    @classmethod
    def create_dataset(cls, dataset_name, bug_id=None, batch_size=256):
        try:
            dataset = cls.get_dataset(dataset_name, bug_id)
            return dataset
        except ValueError:
            # Step1: Initialize the dataset
            dataset_name = dataset_name.upper()
            if dataset_name == 'CIFAR10':
                dataset = CIFAR10Data()
            elif dataset_name == 'ATS':
                dataset = ATSData()
            elif dataset_name == 'CTSD':
                dataset = CTSDData()
            elif dataset_name == 'GTSRB':
                dataset = GTSRBData()
            elif dataset_name == 'TSCD':
                dataset = TSCDData()
            elif dataset_name == 'TTS':
                dataset = TTSData()
            elif dataset_name == 'COVIDX9A':
                dataset = COVIDX9AData()
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")
            # Step2: Inject fault to dataset if bugID is provided
            if bug_id in [7, 8, 9, 10, 11, 12]:
                FaultInjector.inject_dataset_fault(dataset=dataset, bug_id=bug_id)
                dataset.bug_id = bug_id
            else:
                bug_id = None
            # Step3: Packaging the dataloader
            cls.packaging_dataloader(dataset, batch_size=batch_size)
            # Step4: Record the dataset and bug_id in the dataset_list
            key = f"{dataset_name}_{bug_id}" if bug_id else dataset_name
            cls.dataset_list[key] = dataset
            return dataset

    @classmethod
    def get_dataset(cls, dataset_name, bug_id=None):
        # Get the dataset from the dataset_list according to the dataset_name and bugID
        dataset_name = dataset_name.upper()
        key = f"{dataset_name}_{bug_id}" if bug_id in [7, 8, 9, 10, 11, 12] else dataset_name
        if cls.dataset_list.get(key):
            return cls.dataset_list[key]
        else:
            raise ValueError(f"Dataset {key} with bug{bug_id} not registered")
