import torch
import timm
from tqdm import tqdm
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from torch.utils.data import DataLoader
import aimet_torch.quantsim as qsim
from torchvision import transforms, datasets
from aimet_torch.v2.nn import QuantizationMixin
from timm.layers.drop import DropPath
from timm.layers.adaptive_avgmax_pool import FastAdaptiveAvgPool

model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
# model.load_state_dict(torch.load("swin/Swin-Transformer/checkpoints/swin_tiny_patch4_window7_224.pth"))

model.eval()



@QuantizationMixin.implements(DropPath)
class QuantizedDropPath(QuantizationMixin, DropPath):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret
    


@QuantizationMixin.implements(FastAdaptiveAvgPool)
class QuantizedFastAdaptiveAvgPool(QuantizationMixin, FastAdaptiveAvgPool):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, x):
        # Quantize input tensors
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(x)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


# Validation function for PyTorch model with progress bar
def validate_pytorch(model, dataloader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating PyTorch Model", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top5 = outputs.topk(5, dim=1)

            top1_correct += (pred_top1.squeeze() == labels).sum().item()
            top5_correct += sum([labels[i] in pred_top5[i] for i in range(len(labels))])
            total += labels.size(0)

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    return top1_acc, top5_acc

# Define ImageNet validation dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("/media/bmw/datasets/imagenet-1k/val", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Validate model
top1_acc, top5_acc = validate_pytorch(model, dataloader, device)
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")

def apply_aimet_quantization(model, device, dataloader):
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    
    # Print model before BN folding
    print("\nModel before BatchNorm folding:")
    print(model)
    
    fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224), dummy_input=dummy_input)
    
    # Print model after BN folding
    print("\nModel after BatchNorm folding:")
    print(model)
    
    quant_sim = qsim.QuantizationSimModel( 
        model, 
        dummy_input=dummy_input, 
        quant_scheme= 'tf_enhanced',
        rounding_mode= 'nearest',
        default_param_bw= 8,
        default_output_bw = 8
    )
    
    # Use real calibration data for encoding computation
    def forward_pass(model, dataloader):
        model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                print(i , images[0].shape)
                if i >= 10:  # Use only first 10 batches for calibration
                    
                    break
                model(images.to(device))
                # print()
    
    quant_sim.compute_encodings(forward_pass, dataloader)


    return quant_sim.model

quantized_model = apply_aimet_quantization(model,  device, dataloader)

# Validate model
top1_acc, top5_acc = validate_pytorch(quantized_model, dataloader, device)
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
