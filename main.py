import argparse
from src.config_loader import ConfigManager
import os
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run model training with a config file")
parser.add_argument("--config", type=str, required=True, help="Path to the config file")
args = parser.parse_args()

# Load config globally
config = ConfigManager.load_config(args.config)

# Ensure config is loaded
if config is None:
    raise RuntimeError("Config failed to load. Check the config file.")

import torch
from src.model_prep import prepare_and_validate_model
from src.data_loader import load_labeled_data , load_unlabeled_data
from src.auto_quant import run_autoquant
from src.logger import setup_logger
import timm
from src.manual_quantization import manual_quantization
from src.train import train
from src.ptq import cross_layer_equalization


parser = argparse.ArgumentParser(description="Run model training with a config file")
parser.add_argument("--config", type=str, required=True, help="Path to the config file")
args = parser.parse_args()

logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(config["general"]["device"])    
    # Load model
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=config["general"]["pretrained"])
    model.eval()
    model.to(device)

    # Prepare and validate model
    prepared_model = prepare_and_validate_model(model, device)
    


    # Load dataset
    _, dataloader = load_labeled_data(num_classes=config["general"]["num_classes"],images_per_class=config["general"]["images_per_class"], batch_size=config["general"]["batch_size"], is_train=True)
    


    if  config["quantization"]["auto_quant"] == True:
        run_autoquant(prepared_model, dataloader, device)
    if  config["quantization"]["manual_quant"] == True:
        quantized_model = manual_quantization(prepared_model, dataloader, device)
    if  config["quantization"]["qat"]==True:
        quantized_model = manual_quantization(prepared_model, dataloader, device)
        train(quantized_model.model,dataloader=dataloader , device= device ,max_epochs= config['training']['max_epochs'] ,
                learning_rate=config['training']['learning_rate'] ,   
                weight_decay=config['training']['weight_decay'],
                decay_rate=config['training']['decay_rate'],
                learning_rate_schedule=config['training']['learning_rate_schedule'],
                debug_steps=config['training']['debug_steps'])
        logger.info(" Quantization-Aware Training complete.")
        export_path = config["paths"]["save_model_path"] 
        os.makedirs(f"{export_path}/qat_after_train", exist_ok=True)
        quantized_model.export(path=f"{export_path}/qat_after_train", filename_prefix='swin_after_qat',
            dummy_input=torch.randn(1, 3, 224, 224))
    
    if config["quantization"]["ptq_flag"] == True:
        if config["quantization"]["ptq"]['cle'] ==True:
    # Load dataset
            _, dataloader = load_unlabeled_data(num_of_classes=config["general"]["num_classes"],images_per_class=config["general"]["images_per_class"], batch_size=config["general"]["batch_size"])
            cross_layer_equalization(prepared_model , dataloader, device , path= "swin_after_crosslayer")

    
    

    

if __name__ == "__main__":
    main()
