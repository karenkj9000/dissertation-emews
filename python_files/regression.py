#!/usr/bin/env python
# coding: utf-8

# ## 1. Configuration
#
# Set the paths and flags for the modeling process below.
# - `TRAIN_NEW_MODELS`: Set to `True` to run training and tuning. Set to `False` to load existing models.
# - `INPUT_DIR`: The directory where your source data CSV is located.
# - `DATAFRAME_NAME`: The name of your CSV file (without the `.csv` extension).
# - `OUTPUT_DIR_WINDOWS`: The root folder on your Windows drive where model pipelines will be saved. The path is written in WSL format (e.g., `/mnt/d/` for the `D:` drive).

# In[14]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pycaret.regression import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Set to True to run the full training and tuning process.
# Set to False to load pre-existing models from the output directory.
TRAIN_NEW_MODELS = True

# PATHS
INPUT_DIR = os.path.join('..', 'data', 'upsampled')
DATAFRAME_NAME = 'mice_df'  # Name of the .csv file without the extension

# Output directory for saving model pipelines (WSL format for Windows D: drive)
# This path corresponds to D:\ML_Pipelines in Windows
OUTPUT_ROOT_DIR_WINDOWS = '/mnt/d/EMEWS_ML_Pipelines_Output'
DATAFRAME_SPECIFIC_PATH = os.path.join(OUTPUT_ROOT_DIR_WINDOWS, DATAFRAME_NAME)
BASE_MODEL_PATH = os.path.join(DATAFRAME_SPECIFIC_PATH, 'base_models')
TUNED_MODEL_PATH = os.path.join(DATAFRAME_SPECIFIC_PATH, 'tuned_models')

# Create directories if they don't exist
if TRAIN_NEW_MODELS:
    print(f"Creating directory for base models: {BASE_MODEL_PATH}")
    os.makedirs(BASE_MODEL_PATH, exist_ok=True)
    print(f"Creating directory for tuned models: {TUNED_MODEL_PATH}")
    os.makedirs(TUNED_MODEL_PATH, exist_ok=True)
else:
    print("TRAIN_NEW_MODELS is False. Will attempt to load existing models.")


# ## 2. Setup and Data Loading

# In[5]:


# In[9]:


df = pd.read_csv(os.path.join(INPUT_DIR, f'{DATAFRAME_NAME}.csv'))
df['date'] = pd.to_datetime(df['date'])


# ## 3. Pycaret Setup

# In[10]:


fh = 60
split_point = df.shape[0] - fh

train_data = df.iloc[:split_point]
test_data = df.iloc[split_point:]


# In[11]:


setup(
    data=train_data,
    target='total_number_of_patients',
    test_data=test_data,
    fold_strategy='timeseries',
    data_split_shuffle=False,
    fold_shuffle=False,
    session_id=123
)


# ## 4. Model Training or Loading
#
# Based on the `TRAIN_NEW_MODELS` flag, this section will either train and save new models or load existing ones.

# In[15]:


created_models = {}
tuned_models = {}

if TRAIN_NEW_MODELS:
    print("Starting Model Training and Tuning")

    # Step 1: Compare base models
    compare_models(exclude=['lightgbm', 'par', 'dummy', 'lar'], errors='raise')
    best_models_df = pull()
    model_names_to_process = best_models_df.index[:5].tolist()

    # Step 2: Create, save, and tune base models
    for model_name in model_names_to_process:
        print(f'\n--- Processing Model: {model_name} ---')

        # Create base model
        print(f'Creating base model: {model_name}')
        base_model = create_model(model_name, verbose=False)
        created_models[model_name] = base_model

        # Save base model pipeline
        save_path_base = os.path.join(BASE_MODEL_PATH, model_name)
        print(f'Saving base model to: {save_path_base}')
        save_model(base_model, save_path_base)

        # Tune model
        print(f'Tuning model: {model_name}')
        tuned_model = tune_model(
            base_model,
            search_library='scikit-optimize',
            n_iter=20,
            early_stopping=True,
            verbose=False
        )
        tuned_models[model_name] = tuned_model

        # Save tuned model pipeline
        save_path_tuned = os.path.join(TUNED_MODEL_PATH, model_name)
        print(f'Saving tuned model to: {save_path_tuned}')
        save_model(tuned_model, save_path_tuned)

else:
    print("--- Loading Existing Models ---")
    # Load base and tuned models if they exist
    if os.path.exists(BASE_MODEL_PATH):
        model_names_to_process = [os.path.splitext(
            f)[0] for f in os.listdir(BASE_MODEL_PATH) if f.endswith('.pkl')]
        print(f"Found models in {BASE_MODEL_PATH}: {model_names_to_process}")
    else:
        print(
            f"ERROR: Base model directory not found at {BASE_MODEL_PATH}. Cannot load models.")
        model_names_to_process = []

    for name in model_names_to_process:
        base_path = os.path.join(BASE_MODEL_PATH, name)
        tuned_path = os.path.join(TUNED_MODEL_PATH, name)

        # Load Base Model
        if os.path.exists(f'{base_path}.pkl'):
            print(f'Loading base model: {name} from {base_path}')
            created_models[name] = load_model(base_path, verbose=False)
        else:
            print(
                f'WARNING: Base model for {name} not found at {base_path}.pkl')

        # Load Tuned Model
        if os.path.exists(f'{tuned_path}.pkl'):
            print(f'Loading tuned model: {name} from {tuned_path}')
            tuned_models[name] = load_model(tuned_path, verbose=False)
        else:
            print(
                f'WARNING: Tuned model for {name} not found at {tuned_path}.pkl')

print("\nModel processing complete.")
print(f"\nBase models available: {list(created_models.keys())}")
print(f"Tuned models available: {list(tuned_models.keys())}")


# ## 5. Evaluate and Analyze Models
#
# This section provides detailed plots and metrics for each of the models (either newly trained or loaded).

# ### Bayesian Ridge

# In[16]:


# if 'br' in created_models:
#     print("--- Base Bayesian Ridge ---")
#     evaluate_model(created_models['br'])
# if 'br' in tuned_models:
#     print("--- Tuned Bayesian Ridge ---")
#     evaluate_model(tuned_models['br'])


# # ### Lasso Regression

# # In[ ]:


# if 'lasso' in created_models:
#     print("--- Base Lasso Regression ---")
#     evaluate_model(created_models['lasso'])
# if 'lasso' in tuned_models:
#     print("--- Tuned Lasso Regression ---")
#     evaluate_model(tuned_models['lasso'])


# # ### Lasso Least Angle Regression

# # In[ ]:


# if 'llar' in created_models:
#     print("--- Base Lasso Least Angle Regression ---")
#     evaluate_model(created_models['llar'])
# if 'llar' in tuned_models:
#     print("--- Tuned Lasso Least Angle Regression ---")
#     evaluate_model(tuned_models['llar'])


# # ### Elastic Net

# # In[ ]:


# if 'en' in created_models:
#     print("--- Base Elastic Net ---")
#     evaluate_model(created_models['en'])
# if 'en' in tuned_models:
#     print("--- Tuned Elastic Net ---")
#     evaluate_model(tuned_models['en'])


# # ### Huber Regressor

# # In[ ]:


# if 'huber' in created_models:
#     print("--- Base Huber Regressor ---")
#     evaluate_model(created_models['huber'])
# if 'huber' in tuned_models:
#     print("--- Tuned Huber Regressor ---")
#     evaluate_model(tuned_models['huber'])


# ## 6. Custom Metrics and Final Predictions
# This section defines and adds custom metrics for evaluating predictions on the hold-out set, then generates and saves the final performance metrics to an Excel file.

# In[17]:


def r2_rounded(y_true, y_pred):
    """Calculates R2 score after rounding predictions to the nearest whole number."""
    return r2_score(y_true, np.round(y_pred))


def rmse_rounded(y_true, y_pred):
    """Calculates RMSE after rounding predictions to the nearest whole number."""
    return np.sqrt(mean_squared_error(y_true, np.round(y_pred)))


def r2_ceil(y_true, y_pred):
    """Calculates R2 score after ceiling predictions to the nearest whole number."""
    return r2_score(y_true, np.ceil(y_pred))


def rmse_ceil(y_true, y_pred):
    """Calculates RMSE after ceiling predictions to the nearest whole number."""
    return np.sqrt(mean_squared_error(y_true, np.ceil(y_pred)))


def mae_rounded(y_true, y_pred):
    """Calculates MAE after rounding predictions to the nearest whole number."""
    return mean_absolute_error(y_true, np.round(y_pred))


def mae_ceil(y_true, y_pred):
    """Calculates MAE after ceiling predictions to the nearest whole number."""
    return mean_absolute_error(y_true, np.ceil(y_pred))


# In[18]:


try:
    add_metric('R2_Rounded', 'R2_RND', r2_rounded, greater_is_better=True)
    add_metric('RMSE_Rounded', 'RMSE_RND',
               rmse_rounded, greater_is_better=False)
    add_metric('MAE_Rounded', 'MAE_RND', mae_rounded, greater_is_better=False)
    add_metric('R2_Ceil', 'R2_CEIL', r2_ceil, greater_is_better=True)
    add_metric('RMSE_Ceil', 'RMSE_CEIL', rmse_ceil, greater_is_better=False)
    add_metric('MAE_Ceil', 'MAE_CEIL', mae_ceil, greater_is_better=False)
except ValueError:
    print("Metrics may have already been added in this session.")


# In[19]:


# Generate predictions for base models
holdout_predictions_metric = {}
if not created_models:
    print("No base models available to make predictions.")
else:
    for model_name, model_object in created_models.items():
        print(f"Generating predictions for base model: {model_name}")
        predict_model(model_object, verbose=False)
        holdout_predictions_metric[model_name] = pull()

# Generate predictions for tuned models
tuning_predictions_metric = {}
if not tuned_models:
    print("No tuned models available to make predictions.")
else:
    for model_name, model_object in tuned_models.items():
        print(f"Generating predictions for tuned model: {model_name}")
        predict_model(model_object, verbose=False)
        tuning_predictions_metric[model_name] = pull()


# In[26]:


output_excel_path = os.path.join(
    DATAFRAME_SPECIFIC_PATH, 'model_performance_metrics.xlsx')
print(f"Saving performance metrics to: {output_excel_path}")

with pd.ExcelWriter(output_excel_path) as writer:
    # --- Process and Save Base Model Metrics ---
    if holdout_predictions_metric:
        list_of_metric_dfs_base = []
        for model_name, metrics_df in holdout_predictions_metric.items():
            list_of_metric_dfs_base.append(metrics_df)

        results_df_base = pd.concat(
            list_of_metric_dfs_base, ignore_index=True).sort_values('R2', ascending=False)
        print("\n--- Base Model Holdout Predictions ---")
        print(results_df_base.to_string())
        results_df_base.to_excel(
            writer, sheet_name='Base Model Metrics', index=False)
    else:
        print("\nNo base model metrics to save.")

    # --- Process and Save Tuned Model Metrics ---
    if tuning_predictions_metric:
        list_of_metric_dfs_tuned = []
        for model_name, metrics_df in tuning_predictions_metric.items():
            list_of_metric_dfs_tuned.append(metrics_df)

        results_df_tuned = pd.concat(
            list_of_metric_dfs_tuned, ignore_index=True).sort_values('R2', ascending=False)
        print("\n--- Tuned Model Holdout Predictions ---")
        print(results_df_tuned.to_string())
        results_df_tuned.to_excel(
            writer, sheet_name='Tuned Model Metrics', index=False)
    else:
        print("\nNo tuned model metrics to save.")


# ## 7. Final Model Check
#
# As a final step, let's examine the best performing tuned model and its predictions on the test set.

# In[27]:


# Example: Check the tuned Bayesian Ridge model
best_model_name = 'br'

if best_model_name in tuned_models:
    print(f"--- Final Check of Tuned Model: {best_model_name} ---")
    predict_model(tuned_models[best_model_name], verbose=True, data=test_data)
    final_metrics = pull()
    print("\nPerformance on Test Data:")
    print(final_metrics)
else:
    print(
        f"Model '{best_model_name}' not found in the tuned_models dictionary.")
