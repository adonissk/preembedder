**B. Model Architecture & Fixed Training Parameters**

These parameters define the model structure and training settings if HPO is *disabled* (`hpo.enabled: false`) or are used for the *final* training run *after* HPO.

*   `embedding_dim`
    *   **Source:** YAML
    *   **Type:** `int`
    *   **Default (if HPO disabled, `hpo.py`):** `16`
    *   **Description:** Size of the embedding vector for each categorical feature. Note: The model internally applies Layer Normalization to these embeddings after lookup, ensuring the vectors used in the MLP and extracted later have approximately zero mean and unit standard deviation.