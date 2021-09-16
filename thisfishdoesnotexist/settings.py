##########################
### SETTINGS
##########################
import torch

# Device
CUDA_DEVICE_NUM = 0
DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Hyperparameters
RANDOM_SEED = 100
GENERATOR_LEARNING_RATE = 0.0001
DISCRIMINATOR_LEARNING_RATE = 0.00001
WORKERS=2
NUM_EPOCHS = 100
BATCH_SIZE = 64

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 64, 3
