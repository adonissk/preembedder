import json
import os
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom JSON encoder to handle NumPy types if they accidentally appear
# (though the model extraction converts to lists explicitly)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Saves a dictionary to a JSON file.

    Creates the output directory if it doesn't exist.
    Uses a custom encoder to handle potential NumPy types.

    Args:
        data: The dictionary to save.
        filepath: The path to the output JSON file.
        indent: Indentation level for the JSON file.
    """
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, cls=NpEncoder)
        logging.info(f"Data saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to JSON file {filepath}: {e}")
        raise

# Example usage (optional, for testing)
if __name__ == '__main__':
    import numpy as np
    test_data = {
        'model_name': 'preembedder_v1',
        'parameters': {
            'lr': 0.001,
            'layers': [64, 32],
            'numpy_int': np.int64(10),
            'numpy_float': np.float32(0.5)
        },
        'results': {
            'context_A': {
                'score': 0.85,
                'embedding_sample': np.random.rand(5).tolist() # Ensure list for standard JSON
            },
            'context_B': {
                'score': 0.88
            }
        }
    }
    test_filepath = './results/test_output.json' # Save in results dir

    print(f"Saving test data to {test_filepath}...")
    try:
        save_json(test_data, test_filepath)
        print("Test data saved successfully.")

        # Optional: Verify file content
        with open(test_filepath, 'r') as f:
            loaded_data = json.load(f)
            print("File content verified.")
            # print(json.dumps(loaded_data, indent=2))

    except Exception as e:
        print(f"Error during utils test: {e}")
