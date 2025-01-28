from sklearn.metrics import confusion_matrix, average_precision_score

def mAP(labels, scores):


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



def precision_macro(labels, decisions):
    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.
        
        The function outputs the precision macro.
    """
    num_of_classes = precision_macro.c

    # Initialize variables for micro and macro averaging
    precision_macro= 0

    for i in labels:
        binary_labels = (labels == i).astype(int)
        binary_decisions = (decisions == i).astype(int)
        fp,tp = confusion_matrix(binary_labels,binary_decisions).ravel()
        

        # Calculate metrics for the current class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Accumulate for macro-average
        precision_macro += precision
    

    # Calculate macro-averaged metrics
    precision_macro /= num_of_classes
    return precision_macro

def precision_micro(labels, decisions):
    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the precision micro.
    """
    num_of_classes = precision_micro.c

    # Initialize variables for micro and macro averaging
    tp_micro, fp_micro, fn_micro, tn_micro = 0, 0, 0, 0

    for i in labels:
        binary_labels = (labels == i).astype(int)
        binary_decisions = (decisions == i).astype(int)
        tn,fp,fn,tp = confusion_matrix(binary_labels,binary_decisions).ravel()
        
        # Update micro-average counts
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn
        tn_micro += tn


    # Calculate micro-averaged metrics
    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0

    return precision_micro


def recall_macro(labels, decisions):
    num_of_classes = recall_macro.c
    recall_macro = 0

    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the recall macro.
    """

    for i in labels:
        binary_labels = (labels == i).astype(int)
        binary_decisions = (decisions == i).astype(int)

        tn,fp,fn,tp = confusion_matrix(binary_labels,binary_decisions).ravel()

        recall_macro += tp / (tp + fn) if (tp + fn) > 0 else 0

    recall_macro /= num_of_classes
    return recall_macro

def recall_micro(labels, decisions):
    tp_micro, fp_micro, fn_micro, tn_micro = 0, 0, 0, 0
    """
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


        The function outputs the recall micro.
    """


    for i in labels: 
        binary_labels = (labels == i).astype(int)
        binary_decisions = (decisions == i).astype(int)
    
        tn, fp, fn, tp = confusion_matrix(binary_labels, binary_decisions).ravel()

        # Update micro-average counts
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn
        tn_micro += tn

    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
    return recall_micro

def f1_macro(labels, decisions):

    """
    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


    The function outputs the f1 macro.
    """

    num_of_classes = f1_macro.c
    f1_macro = 0

    for i in labels: 
        binary_labels = (labels == i).astype(int)
        binary_decisions = (decisions == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(binary_labels, binary_decisions).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_macro += 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_macro /= num_of_classes
    return f1_macro

def f1_micro(labels, decisions):
    """
    Say that you have classes 0,1,2,3,...,9.
    Then it is assumed that labels and decisions are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
    For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class.


    The function outputs the f1 micro.
    """
    tp_micro, fp_micro, fn_micro, tn_micro = 0, 0, 0, 0

    for i in labels: 
        binary_labels = (labels == i).astype(int)
        binary_decisions = (decisions == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(binary_labels, binary_decisions).ravel()

        # Update micro-average counts
        tp_micro += tp
        fp_micro += fp
        fn_micro += fn
        tn_micro += tn

    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0
    return f1_micro