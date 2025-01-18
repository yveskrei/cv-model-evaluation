import logging

# Model Types
MODEL_DEFAULT = 'DEFAULT'
MODEL_YOLO = 'YOLO'

# Model Tasks
TASK_DETECTION = 'DETECTION'

# Device type
PROVIDER_CPU = 'CPUExecutionProvider'
PROVIDER_CUDA = 'CUDAExecutionProvider'

# COCO classes
SUPPORTED_COCO_CLASSES = [1, 3, 4] # Person, Car, Motorcycle

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)