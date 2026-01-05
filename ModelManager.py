from mymodel import ResNet, AlexNet, DenseNet, VGG16, SqueezeNet
from ModelTrainer import ModelTrainer
from FaultInjector import FaultInjector
import torch
import os


class ModelManager:
    model_list = {}

    @classmethod
    def create_model(cls, model_name, dataset, mode='load', bug_id=None, device=0):
        # Step1. Initialize the model according to the model_name
        model_name = model_name.lower()
        if bug_id in [10, 11, 12]:
            bug_id = None
            if mode == 'train':
                mode = 'load'
        key = f"{model_name}_{dataset.dataset_name}_bug{bug_id}" if bug_id else f"{model_name}_{dataset.dataset_name}"
        if model_name == 'resnet':
            model = ResNet(dataset.num_classes)
        elif model_name == 'alexnet':
            model = AlexNet(dataset.num_classes)
        elif model_name == 'densenet':
            model = DenseNet(dataset.num_classes)
        elif model_name == 'vgg16':
            model = VGG16(dataset.num_classes)
        elif model_name == 'squeezenet':
            model = SqueezeNet(dataset.num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported")
        # Step2. Mark the bugID to the model
        model.bug_id = bug_id if bug_id else None
        # Step3. Inject fault to model if bugID is provided
        if bug_id in [1, 2, 3]:
            FaultInjector.inject_model_fault(model=model, bug_id=bug_id)
        # elif bug_id in [7, 8, 9, 10, 11, 12]:
        #     assert dataset.bug_id == bug_id, f"dataset bug_id {dataset.bug_id} does not match model bugID {bug_id}"
        # Step4. Train or load the model according to the mode
        if mode == 'train':
            cls.__train_model(model=model, dataset=dataset, key=key, device=device)
        elif mode == "evaluate":
            cls.__evaluate_model(model=model, dataset=dataset, key=key, device=device)
        elif mode == 'load':
            cls.__load_model(model=model, key=key, device=device)
        # Step5. Record the model and bug_id in the model_list
        cls.model_list[key] = model

        return model

    @staticmethod
    def __train_model(model, dataset, key, device=0):
        trainer = ModelTrainer(pytorch_model=model.model_structure, device=device)
        if model.bug_id in [4, 5, 6]:
            FaultInjector.inject_hyperparameters_fault(trainer=trainer, bug_id=model.bug_id)
        trainer.train_model(dataset.train_loader, dataset.val_loader)
        model_save_path = os.path.abspath(f"mymodel/model/{key}.pth")
        loss_curve_path = os.path.abspath(f"mymodel/model/loss_curve/{key}.png")
        trainer.save_model(model_save_path=model_save_path)
        trainer.plot_loss_curve(loss_curve_path=loss_curve_path)

    @staticmethod
    def __evaluate_model(model, dataset, key, device=0):
        ModelManager.__load_model(model, key, device=device)
        trainer = ModelTrainer(pytorch_model=model.model_structure, device=device)
        print(f"Evaluating {key} model")
        trainer.evaluate_model(dataset.val_loader)
        trainer.evaluate_model(dataset.test_loader)

    @staticmethod
    def __load_model(model, key, device=0):
        device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.abspath(f"mymodel/model/{key}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.model_structure.load_state_dict(checkpoint)
        model.model_structure.eval()

    @classmethod
    def get_model(cls, model_name, dataset_name, bug_id=None):
        model_name = model_name.lower()
        dataset_name = dataset_name.upper()
        # Get the model from the model_list according to the model_name and bugID
        if bug_id in [10, 11, 12]:
            bug_id = None
        key = f"{model_name}_{dataset_name}_bug{bug_id}" if bug_id else f"{model_name}_{dataset_name}"
        if cls.model_list.get(key):
            return cls.model_list[key]
        else:
            raise ValueError(f"Model {model_name} with bug{bug_id} not registered")
