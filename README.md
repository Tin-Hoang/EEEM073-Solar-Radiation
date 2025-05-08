# EEEM073-Solar-Radiation

A project for forecasting Global Horizontal Irradiance (GHI) using deep learning models based on the National Solar Radiation Database (NSRDB) data.

You can get the experiment runs from below Weights & Biases project:

- Author: Tin Hoang
- Weight & Biases: https://wandb.ai/tin-hoang/EEEM073-Solar-Radiation

## 0. Project Structure

- `data/`: Contains the raw and processed datasets
  - `raw/`: Original data downloaded from NSRDB
  - `processed/`: Preprocessed data ready for model training
- `models/`: Model architecture code for all models implemented in this project.
- `utils/`: Utility functions for data processing and model evaluation, e.g. data loading, training loop, model evaluation, plotting, etc.
- `plots/`: Visualization outputs will be saved here.
- `explainability/`: Outputs of model explainability will be saved here.
- `checkpoints/`: Saved model checkpoints during training will be saved here.

## 1. Installation

See `requirements.txt` for the list of dependencies.

Install the dependencies by running:
```
pip install -r requirements.txt
```

## 2. Download the data

Follow the instructions in `1_data_exploration_and_download.ipynb` to download the raw data, and the `2_data_preprocessing.ipynb` to preprocess the data.

Or, you can download the NSRDB Himawari 7 data at Ho Chi Minh City for 10 years (2011-2020) from the following links:
- Raw data: https://drive.google.com/file/d/1U1RQHxjY50E8aTbF6RBiP08I-kvS6RSN/view?usp=sharing
- Processed data: https://drive.google.com/file/d/1Wjyt_oK4q9au4g6QtTcaxZ3734i5U953/view?usp=sharing

## 3. Notebook Files

This project is organized as a series of Jupyter notebooks that guide you through the entire workflow:

### 1. Data Exploration and Download (`1_data_exploration_and_download.ipynb`)
- Explore the National Solar Radiation Database (NSRDB)
- Download solar radiation data for Ho Chi Minh City, Vietnam (2011-2020)
- Provide data exploration and visualization

### 2. Data Preprocessing (`2_data_preprocessing.ipynb`)
- Load raw NSRDB data
- Select relevant features for modeling
- Create time features and apply data normalization
- Split data into training (2011-2018), validation (2019), and test (2020) sets

### 3. AI Modeling - Basic Approach (`3a_ai_modelling_basic.ipynb`)
- Implement basic models: LSTM, 1D-CNN, CNN-LSTM, MLP, TCN
- Evaluate model performance with various metrics
- Visualize prediction results

### 4. AI Modeling - Advanced Approach (`3b_ai_modelling_advanced.ipynb`)
- Implement more complex deep learning architectures: Transformer, Informer, TSMixer, iTransformer, Mamba
- Evaluate model performance with various metrics
- Visualize prediction results

### 5. Model Explainability (`4_explainability.ipynb`)
- Apply SHAP, and Sensitivity Analysis (gradient-based) to interpret model predictions
- Visualize feature importance
- Provide insights into the factors affecting solar radiation

### 6. Model Compression (`5_model_compression.ipynb`)
- Apply model compression techniques: Quantization, Pruning, and Knowledge Distillation
- Reduce model size while maintaining accuracy
- Evaluate compressed models against original versions

## 4. Alternative: Python Scripts (for running without Jupyter notebooks)

We also provide the Python scripts for each notebook. Each notebook has a corresponding `.py` file with the same name that contains identical code.
These Python scripts allow you to run the workflow without using Jupyter notebooks:

- `1_data_exploration_and_download.py`
- `2_data_preprocessing.py`
- `3a_ai_modelling_basic.py`
- `3b_ai_modelling_advanced.py`
- `4_explainability.py`
- `5_model_compression.py`
