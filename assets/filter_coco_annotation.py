import json

def filter_coco_annotations(input_file, output_file, classes_to_keep):
    """
    Filters a COCO dataset to include only annotations with specified classes.

    Args:
        input_file (str): Path to the input COCO JSON file.
        output_file (str): Path to save the filtered COCO JSON file.
        classes_to_keep (list): List of class IDs or names to keep in the dataset.
    """
    # Load the COCO dataset
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    # Get category IDs to keep
    category_ids_to_keep = []
    if isinstance(classes_to_keep[0], str):  # If classes_to_keep are names
        for category in coco_data['categories']:
            if category['name'] in classes_to_keep:
                category_ids_to_keep.append(category['id'])
    else:  # If classes_to_keep are IDs
        category_ids_to_keep = classes_to_keep

    # Filter annotations
    filtered_annotations = [
        annotation for annotation in coco_data['annotations']
        if annotation['category_id'] in category_ids_to_keep
    ]

    # Filter images based on the remaining annotations
    annotation_image_ids = {annotation['image_id'] for annotation in filtered_annotations}
    filtered_images = [
        image for image in coco_data['images']
        if image['id'] in annotation_image_ids
    ]

    # Filter categories
    filtered_categories = [
        category for category in coco_data['categories']
        if category['id'] in category_ids_to_keep
    ]

    # Create the filtered COCO dataset
    filtered_coco_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    # Save the filtered dataset
    with open(output_file, 'w') as f:
        json.dump(filtered_coco_data, f, indent=4)

    print(f"Filtered dataset saved to {output_file}")

# Example usage
input_coco_file = 'instances_val2017.json'
output_coco_file = 'annotation.json'
classes_to_keep = [1, 3, 4]

filter_coco_annotations(input_coco_file, output_coco_file, classes_to_keep)
