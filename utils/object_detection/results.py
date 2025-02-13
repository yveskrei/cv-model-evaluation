import logging
import torch
import json
import os

# Variables
logger = logging.getLogger(__name__)

def get_torchmetrics_results(results_path: str):
    """
        Loads results from json file(torchmetrics format)
        Converts all dicts into torch tensors
    """
    if not os.path.exists(results_path):
        raise Exception(f"Results file not found at {results_path}")
    try:
        # Read results from file
        with open(results_path, 'r', encoding='utf-8') as file:
            inference_results = json.load(file)
        
        # Convert dicts of values into torch tensors
        inference_results = get_results_tensors(inference_results)
        
        return inference_results
    except Exception as e:
        logger.error(e)
        raise Exception('Could not load results from file')
    

def get_results_json(results: list[dict]):
    """
        Format inference results into json type, so we can
        export those to json file
    """

    # Convert tensors into lists recursively
    if isinstance(results, torch.Tensor):
        return results.tolist()
    elif isinstance(results, dict):
        return {key: get_results_json(value) for key, value in results.items()}
    elif isinstance(results, list):
        return [get_results_json(item) for item in results]
    else:
        return results
    
def get_results_tensors(results: list[dict]):
    """
        Format inference results from json to tensors,
        so we can use them in code
    """
    if isinstance(results, list):  
        return torch.tensor(results) if all(isinstance(i, (int, float, list)) for i in results) else [get_results_tensors(i) for i in results]
    elif isinstance(results, dict):  
        return {key: get_results_tensors(value) for key, value in results.items()}  
    else:
        return results  # Return unchanged if not a list/dict