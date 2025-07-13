import argparse
import json
import logging
import os

# Custom modules
import utils.config as config
import utils.object_detection.evaluator as object_detection

# Variables
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="Inference - Path to the model file")
    parser.add_argument("--model_type", type=str, default=config.MODEL_YOLO, help="Inference - Type of the model (e.g., 'yolo')")
    parser.add_argument("--coco_annotation", type=str, default=None, help ="Inference - Path to the COCO annotation file")
    parser.add_argument("--results_path", type=str, default=None, help="Evaluation - Path to the results file")

    args = parser.parse_args()
    eval_results = None

    # Create results directory
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Check if able to do inference
    if args.model_path and args.coco_annotation:
        model = None

        # Decide model type based on file extension
        if args.model_path.endswith('.onnx'):
            from utils.wrapper_onnx import Wrapper

            # Initialize the ONNX wrapper
            model = Wrapper(
                model_path=args.model_path,
                model_type=args.model_type
            )

        elif args.model_path.endswith('.trt') or args.model_path.endswith('.engine') or args.model_path.endswith('.plan'):
            from utils.wrapper_trt import Wrapper

            # Initialize the TensorRT wrapper
            model = Wrapper(
                model_path=args.model_path,
                model_type=args.model_type
            )
        else:
            raise Exception("Unsupported model file type. Use .onnx or .trt files.")

        inf_results, eval_results = object_detection.evaluate_model_inference(
            model=model,
            coco_annotation=args.coco_annotation
        )

        # Save inference results to JSON
        inf_results_path = os.path.join(results_dir, 'inference_results.json')
        with open(inf_results_path, 'w', encoding='utf-8') as file:
            json.dump(inf_results, file, indent=4)

    elif args.results_path:
        # Evaluate results from a file
        eval_results = object_detection.evaluate_results_file(
            results_path=args.results_path
        )
    else:
        raise Exception("Unable to provide sufficient arguments for inference or evaluation.")
    
    # Save results to JSON
    if eval_results:
        eval_results_path = os.path.join(results_dir, 'eval_results.json')

        with open(eval_results_path, 'w') as file:
            json.dump(eval_results, file, indent=4)
        
        logger.info(f"Evaluation results saved to {eval_results_path}")
    else:
        logger.warning("No evaluation results to save.")