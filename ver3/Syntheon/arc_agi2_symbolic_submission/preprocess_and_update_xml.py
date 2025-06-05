import xml.etree.ElementTree as ET

def preprocess_arc_agi_task(task):
    """Preprocess an arc_agi_task element to add preprocessing information."""
    metadata = task.find('metadata')
    if metadata is None:
        return

    # Example preprocessing: Add a new element for preprocessing
    preprocessing = ET.SubElement(metadata, 'preprocessing')

    # Add transformation hints as preprocessing steps
    transformations = metadata.find('statistics/transformations')
    if transformations is not None:
        for transformation in transformations.findall('transformation'):
            hint = transformation.text
            step = ET.SubElement(preprocessing, 'step')
            step.text = f"Apply rule based on hint: {hint}"

    # Add KWIC analysis as preprocessing steps
    kwic = metadata.find('kwic')
    if kwic is not None:
        for pair in kwic.findall('pair'):
            color1 = pair.get('color1')
            color2 = pair.get('color2')
            frequency = pair.get('frequency')
            step = ET.SubElement(preprocessing, 'step')
            step.text = f"Analyze color pair ({color1}, {color2}) with frequency {frequency}"

def update_xml_with_preprocessing(input_path, output_path):
    """Update the XML file with preprocessing information."""
    tree = ET.parse(input_path)
    root = tree.getroot()

    for task in root.findall('arc_agi_task'):
        preprocess_arc_agi_task(task)

    tree.write(output_path, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    input_path = "input/arc_agi2_training_combined.xml"
    output_path = "input/arc_agi2_training_combined_preprocessed.xml"
    update_xml_with_preprocessing(input_path, output_path)
    print(f"Preprocessed XML saved to {output_path}")
