## 🔧 Dependencies
The code has been tested under the following environment:  

| Package           | Version   |
|-------------------|-----------|
| Python            | 3.12.4    |
| PyTorch           | 2.7.0     |
| CUDA              | 12.6      |
| transformers      | 4.30.0    |


## 🚀 Quick Start

### 1. Data Processing Pipeline
Execute the following scripts in sequence to preprocess the data and run the model:

1.  **Preprocessing**: Run `Predata.py` to prepare the raw data.
    ```bash
    python Predata.py
    ```

2.  **Splitting Datasets**: Use `split_name.py` to partition the data into training, validation, and test sets.
    ```bash
    python split_name.py
    ```

3.  **Model Training & Evaluation**: Finally, execute `main.py` to start the training process.
    ```bash
    python main.py
    ```

### 2. Configuration
Before running the main script, please set the following variables in your code to configure the data protocol and save paths:
```python
# Define your data protocol/selection setting
data_select = "D_to_D"
setting = "D_to_D"

# Define output paths (AUC results and model save location)
file_AUCs = 'output/result/AUCs--' + setting + '.txt'
file_model = 'output/model/' + setting
