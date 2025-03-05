
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.logger import setup_logger
from src.config_loader import ConfigManager

config = ConfigManager.get_config()
logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])




TRAIN_DIR = config['paths']["train_data_path"]
VAL_DIR = config['paths']["val_data_path"]

def load_labeled_data(num_classes=50, images_per_class=10, batch_size=128, is_train=False):
    logger.info("Loading labeled dataset...")


    
    transform_data = transforms.Compose([
        transforms.Resize(249, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675/255, 116.28/255, 103.53/255], std=[58.395/255, 57.12/255, 57.375/255])
    ])

    data_dir = TRAIN_DIR if is_train else VAL_DIR
    print(f"Loading data from: {data_dir}")
    dataset = ImageFolder(root=data_dir, transform=transform_data)


    # Filter dataset to include only selected classes
    selected_classes = dataset.classes[:num_classes]
    selected_class_indices = [dataset.class_to_idx[class_name] for class_name in selected_classes]

    selected_indices = []
    class_counts = {idx: 0 for idx in selected_class_indices}

    for idx, (_, label) in enumerate(dataset.samples):
        if label in selected_class_indices and class_counts[label] < images_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1

    subset_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    logger.info(f"Dataset size: {len(subset_dataset)} | DataLoader batches: {len(dataloader)}")
    
    return subset_dataset, dataloader




def load_unlabeled_data(num_of_classes:int =50, images_per_class:int=10,batch_size:int =128) -> DataLoader :
    
    
    dataset,dataloader =load_labeled_data(num_of_classes,images_per_class,batch_size)

    class UnlabelledDataset(ImageFolder):
        def __init__(self, dataset):
            self._dataset = dataset

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, index):
            images, _ = self._dataset[index]
            return images
        
    unlabelled_dataset=UnlabelledDataset(dataset)
    
    unlabelled_dataloader= DataLoader (
    unlabelled_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory= True
    )
    
    return unlabelled_dataset,unlabelled_dataloader

