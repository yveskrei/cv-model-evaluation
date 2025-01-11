from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def plot_image_predictions(image: Image, annotations: list, predictions: list):
    """
        Plots image with annotations and predictions
    """
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Plot annotations
    for annotation in annotations:
        bbox = annotation["bbox"]
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.3
        )
        ax.add_patch(rect)

    # Plot predictions
    for prediction in predictions:
        bbox = prediction["bbox"]
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=2,
            edgecolor="blue",
            facecolor="blue",
            alpha=0.25
        )
        ax.add_patch(rect)

        ax.text(
            bbox[0] + bbox[2], 
            bbox[1] + bbox[3] + 10, 
            f"Class: {prediction['class_id']}, Score: {prediction['score']:.2f}", 
            color='blue', 
            fontsize=11, 
            weight='bold'
        )

    plt.show()

def plot_pr_graph(results: list):
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        thresholds = [r["confidence_threshold"] for r in results]

        plt.figure()
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()