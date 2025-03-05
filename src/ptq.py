from aimet_torch.cross_layer_equalization import equalize_model
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantParams
from aimet_torch.bias_correction import correct_bias
from src.manual_quantization import manual_quantization
from src.config_loader import ConfigManager
from src.logger import setup_logger
import torch
config = ConfigManager.get_config()


logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])


def cross_layer_equalization(model , dataloader , device , path=None):
    equalize_model(model, input_shapes=(1, 3, 224, 224))

    if config['quantization']['ptq']['bc']==True and any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules()):
        choosen = QuantScheme.post_training_tf
        if  config['quantization']['quant_scheme'] == 'tf_enhanced':
            choosen = QuantScheme.post_training_tf_enhanced
        
        bc_params = QuantParams(weight_bw= config['quantization']['weight_bw'], act_bw=config['quantization']['act_bw'], round_mode=config['quantization']['round_mode'],
                                quant_scheme=choosen)
        
        correct_bias(model, bc_params, num_quant_samples=16,
                     data_loader=dataloader, num_bias_correct_samples=16)
    else:
        print("Skipping bias correction as no BatchNorm layers were found.")
        
    manual_quantization(model ,dataloader , device , path)
    
    




