# Calculated metrics
- **ROC Curve** - shows TRP/FPR under different **confidence** intervals. shows how good model classifies. below 0.5 - no difference from random guessing.
- **P/R Curve** - shows Precision/Recall under different **confidence** intervals. Similar to ROC, good for imbalanced datasets.
<br>
<img src="images/pr_roc_curves.png" width="600" height="300" alt="PR/ROC Curves">
- **Screw up images** - shows images with bad IOU's between ground truth and predictions from the model. Helps evaluating errors.
<br>
<img src="images/iou_images.jpg" width="600" height="300" alt="PR/ROC Curves">