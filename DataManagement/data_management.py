from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class DataManagement:

    def __init__(self, data_path):
        self.data_path = data_path

        self.batch_size = 2  # Default Setting
        self.data_num = 0

        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_dataset = datasets.ImageFolder(
            data_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization
            ])
        )

        self.data_num = len(self.image_dataset)

    def get_loader(self, shuffle=True, num_worker=0):
        loader = DataLoader(
            self.image_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_worker,
            pin_memory=True,
            sampler=None
        )
        return loader

    def set_batch_size(self, batch_size=2):
        self.batch_size = batch_size
