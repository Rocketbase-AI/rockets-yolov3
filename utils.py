from operator import itemgetter
import os
import json

def orderAnchors(list_anchors: list) -> list:
    """Order the anchors from the biggest width to the smallest width.

    Args:
        list_anchors: list of anchors with the following structure: [[w,h],...]

    Returns:
        List of anchors ordered from the biggest width to the smallest width.
    """
    return sorted(list_anchors, key = itemgetter(0), reverse = True)

def importConfig(path_config: str) -> dict:
    """Import the .json configuration file into a dict

    Args:
        path_config: str to the .json config file
    
    Returns:
        Dict of all the configuration parameters
    """
    assert os.path.exists(path_config), "Impossible to find " + path_config

    with open(path_config) as f:
        config = json.load(f)

    # Check the information in the config file
    assert 'anchors' in config.keys(), "Missing the anchors dimensions -> config['anchors'] = [[width, height], ...x9]"
    assert len(config['anchors']) == 9, "9 anchors are needed: " + str(len(config['anchors'])) + " found."
    
    assert 'num_classes' in config.keys(), "Missing the number of classes -> config['num_classes'] = int(num_classes)"
    assert config['num_classes'] > 0, "The number of classes need to be positive."

    assert 'input' in config.keys(), "Missing the dimensions of the input -> config['input'] = [width, height, channels]"
    assert min(config['input']) > 0, "The dimensions of the input need to be positive."

    # Order the anchors
    config['anchors'] = orderAnchors(config['anchors'])

    # Get img_size from the input dimensions
    config['img_size'] = config['input'][0]

    return config
