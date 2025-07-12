import logging

# Model Types
MODEL_YOLO = 'YOLO'

# Model Tasks
TASK_DETECTION = 'DETECTION'

# Device type
PROVIDER_CPU = 'CPUExecutionProvider'
PROVIDER_CUDA = 'CUDAExecutionProvider'

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)