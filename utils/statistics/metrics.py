from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
from object_detection import get_torchmetrics_annotations, get_torchmetrics_predictions
import torch, torchvision





def auc_roc(labels,scores): # Area Under Curve of !!!!! ROC GRAPH !!!!!
    """
    scores is a big tensor/nparray of probabilities, for all classes per data sample (data sample = bbox)

    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.
    In this case its binary so labels should be 0s and 1s.
    """
    return roc_auc_score(y_true = labels, y_score = scores)


def auc_prc(labels,scores): #Area Under Curve of !!!!!!!! PRC GRAPH !!!!!!
    """
    scores is a big tensor/nparray of probabilities, for all classes per data sample (data sample = bbox)

    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.
    In this case its binary so labels should be 0s and 1s.
    """
    return average_precision_score(y_true=labels, y_score=scores)

    
def mAP(labels, scores): # Average Of Areas Under Curve of !!!!!!PRC GRAPH!!!!!


    """
    scores is a big tensor/nparray of probabilities, for all classes per data sample (data sample = bbox)

    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.
    """
    num_of_classes = mAP.c
    APs = []
    for i in range(num_of_classes):
        classLabels = [1 if label == i else 0 for label in labels]
        y_scores_class_i = scores[:,i]

        APs.append(average_precision_score(y_true=classLabels,y_score=y_scores_class_i))
    return sum(APs)/len(APs)


def precision_binary(labels, predictions):
    """

        The function outputs the precision.

    """

    threshold = precision_binary.threshold
    tp, fp, fn = get_confusion_matrix(predictions=predictions, annotations=labels,confidence_threshold= threshold)
    return tp/(tp+fp) if tp+fp>0 else 0 
        
    

def precision_multiclass_macro(labels, predictions):
    """
        The function outputs the precision macro.

        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.
    """

    num_of_classes = precision_multiclass_macro.c
    threshold = precision_multiclass_macro.threshold
    if not isinstance(threshold,list):
        threshold = [threshold for i in range(num_of_classes)]
    
    precision_macro= 0

    
    for i in range(num_of_classes):
        results = get_torchmetrics_predictions(predictions,threshold[i])
        classLabels = [1 if label == i else 0 for label in labels]
        tp, fp, fn = get_confusion_matrix(classLabels, results[only class = i], iou_thresh=0.5)


        # Calculate metrics for the current class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Accumulate for macro-average
        precision_macro += precision
    

    # Calculate macro-averaged metrics
    precision_macro /= num_of_classes
    return precision_macro

def precision_multiclass_micro(labels, decisions):
    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the precision micro.
    """
    num_of_classes = precision_micro.c

    # Initialize variables for micro and macro averaging
    tp_micro, fp_micro, fn_micro, tn_micro = 0, 0, 0, 0

    threshold = precision_multiclass_micro.threshold

    if not isinstance(threshold,list):
        threshold = [threshold for i in range(num_of_classes)]

    
    for i in range(num_of_classes):
        results = get_torchmetrics_predictions(predictions,threshold[i])
        classLabels = [1 if label == i else 0 for label in labels]
        tp, fp, fn = get_confusion_matrix(classLabels, results[only class = i], iou_thresh=0.5)
        
        # Update micro-average counts
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn


    # Calculate micro-averaged metrics
    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0

    return precision_micro

def recall_binary(labels, predictions):
    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the recall macro.
    """


    threshold = recall_binary.threshold
    tp, fp, fn = get_confusion_matrix(predictions=predictions, annotations=labels,confidence_threshold= threshold)
    return tp/(tp+fn) if tp+fn>0 else 0 



def recall_multiclass_macro(labels, decisions):

    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the recall macro.
    """
    num_of_classes = recall_macro.c
    recall_macro = 0
    threshold = recall_multiclass_macro.threshold
    if not isinstance(threshold,list):
        threshold = [threshold for i in range(num_of_classes)]

    
    for i in range(num_of_classes):
        results = get_torchmetrics_predictions(predictions,threshold[i])
        classLabels = [1 if label == i else 0 for label in labels]
        tp, fp, fn = get_confusion_matrix(classLabels, results[only class = i], iou_thresh=0.5)


        recall_macro += tp / (tp + fn) if (tp + fn) > 0 else 0

    recall_macro /= num_of_classes
    return recall_macro

def recall_multiclass_micro(labels, decisions):
    tp_micro, fp_micro, fn_micro, tn_micro = 0, 0, 0, 0
    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the recall micro.
    """

    num_of_classes = recall_micro.c
    tp_micro,fp_micro,fn_micro,tn_micro = 0,0,0,0


    threshold = recall_multiclass_micro.threshold
    if not isinstance(threshold,list):
        threshold = [threshold for i in range(num_of_classes)]

    
    for i in range(num_of_classes):
        results = get_torchmetrics_predictions(predictions,threshold[i])
        classLabels = [1 if label == i else 0 for label in labels]
        tp, fp, fn = get_confusion_matrix(classLabels, results[only class = i], iou_thresh=0.5)

        tp_micro += tp
        fp_micro += fp
        fn_micro += fn

    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
    return recall_micro

def f1_binary(labels,decisions):
    """
    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


    The function outputs the f1 macro.
    """
    threshold = f1_binary.threshold
    results = get_torchmetrics_predictions(predictions,threshold)
    tp, fp, fn = get_confusion_matrix(classLabels, results,0.5)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def f1_multiclass_macro(labels, decisions):

    """
    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


    The function outputs the f1 macro.
    """

    num_of_classes = f1_multiclass_macro.c
    f1_multiclass_macro = 0
    threshold = f1_multiclass_macro.threshold
    results = get_torchmetrics_predictions(predictions,threshold)
    if not isinstance(threshold,list):
        threshold = [threshold for i in range(num_of_classes)]


    for i in range(num_of_classes):
        results = get_torchmetrics_predictions(predictions,threshold[i])
        classLabels = [1 if label == i else 0 for label in labels]
        tp, fp, fn = get_confusion_matrix(classLabels, results[only in class i], iou_thresh=0.5)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_multiclass_macro += f1
    f1_multiclass_macro /= num_of_classes
    return f1_multiclass_macro

def f1_multiclass_micro(labels, decisions):
    """
    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


    The function outputs the f1 micro.
    """
    f1_micro = 0
    num_of_classes = f1_multiclass_micro.c

    tp_micro, fp_micro, fn_micro, tn_micro = 0, 0, 0, 0

    threshold = f1_multiclass_micro.threshold
    
    if not isinstance(threshold,list):
        threshold = [threshold for i in range(num_of_classes)]

    for i in range(num_of_classes):
        results = get_torchmetrics_predictions(predictions,threshold[i])
        classLabels = [1 if label == i else 0 for label in labels]
        tp, fp, fn = get_confusion_matrix(classLabels, results[only in class i], iou_thresh=0.5)

        # Update micro-average counts
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn

    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    return f1_micro


import numpy as np

print(np.array([1,2,3]))