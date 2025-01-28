import numpy as np
import torch

# Custom modules
from utils.statistics.bootstrap import Bootstrap
from utils.statistics.metrics import mAP, precision_macro, precision_micro, recall_macro, recall_micro, f1_macro, f1_micro

# Variables
METRIC_MAP = 'mAP'
METRIC_PRECISION_MACRO = 'precision_macro'
METRIC_PRECISION_MICRO = 'precision_micro'
METRIC_RECALL_MACRO = 'recall_macro'
METRIC_RECALL_MICRO = 'recall_micro'
METRIC_F1_MACRO = 'f1_macro'
METRIC_F1_MICRO = 'f1_micro'
METRIC_TP = 'tp'
METRIC_FP = 'fp'
METRIC_FN = 'fn'



def process_inference(data: torch.Tensor | np.ndarray | list):
    if not isinstance(data, (torch.Tensor, list, np.array)):
        raise Exception("Data type is not Tensor or numpy array or list. Please enter one of those.")

def compute_confidence_interval(samples, metric,number_of_classes, labels=None, conditions=None, num_bootstraps=1000, alpha=5, samples2=None):
    """
    Documentation
    """
    process_inference(samples) # Raise exception if needed for inference scores.
    process_inference(labels) # Raise exception if needed for labels.

    #tp, fp, fn = self.get_confusion_matrix(filtered_predictions, annotations, IOU_THRESHOLD, conf_threshold)

    if metric == METRIC_MAP:
        return evaluate_with_conf_int(samples=samples, metric=mAP, number_of_classes=number_of_classes, labels=labels, conditions=conditions, num_bootstraps=num_bootstraps,alpha=alpha,samples2=samples2)
    # elif metric == METRIC_PRECISION:
    #     return
    #     #return evaluate_with_conf_int(samples=samples, metric=, number_of_classes=number_of_classes, labels=labels, conditions=conditions, num_bootstraps=num_bootstraps,alpha=alpha,samples2=samples2)
    # elif metric == METRIC_RECALL:
    #     return
    # elif metric == METRIC_F1:
    #     return
    # elif metric == METRIC_ACCURACY:
    #     return

def evaluate_with_conf_int(samples, metric,number_of_classes, labels=None, conditions=None, num_bootstraps=1000, alpha=5, samples2=None):
    """
    Supports both
    """

    """ Evaluate the metric on the provided data and then run bootstrapping to get a confidence interval.
        
        - samples: array of decisions/scores/losses for each sample needed to compute the metric.
                
        - metric: function that computes the metric given a set of samples. The function will be 
        called internally as metric([labels], samples, [samples2]), where the two arguments in 
        brackets are optional (if they are None, they are excluded from the call). 
        
        - labels: array of labels or any per-sample value needed to compute the metric. May be None
        if the metric can be computed just with the values available in the samples array. 
        Default=None.

        - conditions: integer array indicating the condition of each sample (in the same order as
        labels and samples). Default=None.
        
        - num_bootstraps: number of bootstraps sets to create. Default=1000.
        
        - alpha: the confidence interval will be computed between alpha/2 and 100-alpha/2 
        percentiles. Default=5.
        
        - samples2: second set of samples for metrics that require an extra input. Default=None.

        See https://github.com/luferrer/ConfidenceIntervals for more details. 
    """
    if torch.is_tensor(samples):
        samples = np.array(samples.tolist())
    elif isinstance(samples,list):
        samples = np.array(samples)
    if torch.is_tensor(labels):
        labels = np.array(labels.tolist())
    elif isinstance(labels, list):
        labels = np.array(labels) 
    metric.c = number_of_classes
    center = Bootstrap.metric_wrapper(labels, samples, samples2,metric)
    bt = Bootstrap(num_bootstraps, metric)
    ci = bt.get_conf_int(samples, labels, conditions, alpha=alpha, samples2=samples2)
    
    return center, ci