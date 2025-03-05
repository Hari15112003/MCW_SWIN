import torch
from aimet_torch.model_preparer import prepare_model
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.meta.connectedgraph import ConnectedGraph
from src.config_loader import ConfigManager
from src.logger import setup_logger

config = ConfigManager.get_config()  # Access global config safely

logger = setup_logger(config["paths"]["log_file"], config["logging"]["log_level"])



def prepare_and_validate_model(model, device):
    logger.info("Preparing and validating the model for quantization...")

    def prepare_and_check(model):
        prepared_model = prepare_model(model)
        invalid_layers = ModelValidator.validate_model(prepared_model, model_input=torch.randn(1, 3, 224, 224).to(device))
        return prepared_model, invalid_layers

    # First attempt
    prepared_model, invalid_layers = prepare_and_check(model)

    if invalid_layers:
        logger.error("âŒ Model contains unsupported layers for AIMET quantization.")
        logger.info("Adding additional operations and reattempting preparation...")

        # Extend AIMET ConnectedGraph with additional operations
        additional_ops = [
            "floor_divide", "remainder", "pad", "dropout", "Add", "linear", "pow",
            "Mul", "matmul", "softmax", "rsub", "roll", "fill", "masked_fill",
            "new_zeros", "sub", "Concat"
        ]
        ConnectedGraph.math_invariant_types.update(additional_ops)

        # Second attempt after adding operations
        prepared_model, invalid_layers = prepare_and_check(model)

        if invalid_layers:
            logger.error("ðŸš¨ Model still contains unsupported layers after adding operations. Exiting.")
            return None

    logger.info("âœ… Model successfully prepared and validated for AIMET quantization.")
    return prepared_model
