import torch
from aimet_torch.auto_quant import AutoQuant
from aimet_torch.adaround.adaround_weight import AdaroundParameters
from src.data_loader import  load_unlabeled_data
from src.logger import setup_logger
from src.eval import eval_callback
from src.config_loader import ConfigManager
import os

config = ConfigManager.get_config()
logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])

def run_autoquant(prepared_model, dataloader, device):
    """
    Runs AIMET AutoQuant on the prepared model.
    """
    logger.info("?? Running AutoQuant...")

    dummy_input = torch.rand(1, 3, 224, 224).to(device)
    dataset,dataloader =load_unlabeled_data(20,20,32)
    export_path = config["paths"]["save_model_path"] 
    os.makedirs(f"{export_path}/auto_quant", exist_ok=True)
    auto_quant = AutoQuant(prepared_model, dummy_input=dummy_input, data_loader=dataloader,
                           eval_callback=eval_callback, results_dir=f"{export_path}/auto_quant", model_prepare_required=False)

    # Run inference before optimization
    sim, initial_accuracy = auto_quant.run_inference()
    logger.info("- Initial Quantized Accuracy: {initial_accuracy:.4f}")

    # Set AdaRound parameters
    adaround_params = AdaroundParameters(dataloader, num_batches=len(dataloader), default_num_iterations= config['auto_quant']['default_num_iterations'])
    auto_quant.set_adaround_params(adaround_params)

    # Optimize the model
    logger.info("?? Optimizing model with AutoQuant...")
    
    result = auto_quant.optimize(allowed_accuracy_drop=config['auto_quant']['allowed_accuracy_drop'])

    if len(result) == 3:
        model, optimized_accuracy, encoding_path = result  # Only 3 values returned
        pareto_front = None  # Set to None if not returned
    else:
        model, optimized_accuracy, encoding_path, pareto_front = result  # If all 4 are returned
    

    

    
    logger.info("Optimized Quantized Accuracy: {optimized_accuracy:.4f}")

    return model, optimized_accuracy, encoding_path
