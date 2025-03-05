import torch
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from src.eval import eval_callback
from src.logger import setup_logger
import os
from src.data_loader import load_labeled_data
from src.config_loader import ConfigManager

config = ConfigManager.get_config()

logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])


def pass_calibration_data(model, dataloader, device):
    batch_size = dataloader.batch_size
    model.eval()
    samples = 1000  # Limit calibration samples
    batch_cntr = 0
    with torch.no_grad():
        
        for i, input_data in enumerate(dataloader):

            # inputs_batch = input_data.to(device)
            inputs_batch = input_data[0].to(device) if isinstance(input_data, list) else input_data.to(device)

            model(inputs_batch)  # Forward pass
            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break



def manual_quantization(prepared_model, dataloader, device , path =None ):
    logger.info("Performing Manual Quantization...")
    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    choosen = QuantScheme.post_training_tf
    if  config['quantization']['quant_scheme'] == 'tf_enhanced':
        choosen = QuantScheme.post_training_tf_enhanced
    sim = QuantizationSimModel(
        model=prepared_model,
        quant_scheme=choosen,
        dummy_input=dummy_input,
        default_output_bw= config['quantization']['default_output_bw'],
        default_param_bw= config['quantization']['default_param_bw']
    )
    
    _, dataloader = load_labeled_data(num_classes=config["general"]["num_classes"],images_per_class=config["general"]["images_per_class"], batch_size=config["general"]["batch_size"], is_train=True)
    
    sim.compute_encodings(forward_pass_callback=lambda model: pass_calibration_data(model, dataloader, device))
    accuracy = eval_callback(sim.model)
    logger.info(f"Manual Quantization Accuracy: {accuracy:.4f}")
    export_path = config["paths"]["save_model_path"] 
    if path == None :
        os.makedirs(f"{export_path}/manual_qat", exist_ok=True)
        sim.export(path=f"{export_path}/manual_qat", filename_prefix='swin_after_qat',
            dummy_input=torch.randn(1, 3, 224, 224))
    else:
        os.makedirs(f"{export_path}/cross_layer", exist_ok=True)
        sim.export(path=f"{export_path}/cross_layer", filename_prefix='swin_after_crosslayer',
            dummy_input=torch.randn(1, 3, 224, 224))

    return sim