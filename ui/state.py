from core.model_manager import ModelManager
from core.lora_manager import LoraManager
from core.generation import GenerationCoordinator

# Initialize the global backend managers
model_manager = ModelManager()
lora_manager = LoraManager()
coordinator = GenerationCoordinator(model_manager=model_manager)