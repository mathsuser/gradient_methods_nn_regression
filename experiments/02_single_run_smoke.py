# The goal of this script is to run a single experiment with a small number of iterations to ensure that the code runs without errors. 
# It is not intended to produce meaningful results, but rather to serve as a smoke test for the experiment pipeline.


from pathlib import Path
import json
import numpy as np
import torch
import copy

from gradient_methods_nn_regression.model import TinyRegressionModel
from gradient_methods_nn_regression.training import train_model


# 1. Load configuration files:
# baseline_cfg defines the fixed data/model setup, while experiment_cfg defines
# the named sampling methods used by the Week 1 comparison.
config_path = Path("configs/baseline.json")
experiment_path = Path("configs/experiments/week1_gradient_methods.json")

with config_path.open("r", encoding="utf-8") as handle:
    baseline_cfg = json.load(handle)

with experiment_path.open("r", encoding="utf-8") as handle:
    experiment_cfg = json.load(handle)

# 2. Load the dataset: Train dataset + Validation dataset

data_path = Path(baseline_cfg["paths"]["generated_data_dir"])

# 2.1 Load the training dataset

with np.load(data_path / "baseline_train.npz") as data:
    X_train = data["x"].astype(np.float32, copy=False)
    y_train = data["y"].astype(np.float32, copy=False).reshape(-1, 1)  # Ensure y_train is a column vector
    # f_true is the noiseless target function value; it lets us measure
    # function-estimation error separately from noisy prediction error.
    f_true = data["f_true"].astype(np.float32, copy=False).reshape(-1, 1)  # Ensure f_true is a column vector

    # Convert to torch from numpy
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    f_true = torch.from_numpy(f_true)


print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"f_true shape: {f_true.shape}")

# 2.2 Load the validation dataset
with np.load(data_path / "baseline_validation.npz") as data:
    X_val = data["x"].astype(np.float32, copy=False)
    y_val = data["y"].astype(np.float32, copy=False).reshape(-1, 1)  # Ensure y_val is a column vector
    # Validation keeps both noisy y and noiseless f_true for separate metrics.
    f_val_true = data["f_true"].astype(np.float32, copy=False).reshape(-1, 1)  # Ensure f_val_true is a column vector

    # Convert to torch from numpy
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)  
    f_val_true = torch.from_numpy(f_val_true)


print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"f_val_true shape: {f_val_true.shape}")



# 3. Instantiate the model
# This seed controls the model's random initial weights.
seed = torch.manual_seed(0)  # Set the random seed for reproducibility
model = TinyRegressionModel()
initial_state = copy.deepcopy(model.state_dict())  # Save the initial state of the model for later comparison


# 4. Train the model

# 4.1 Set the training parameters

# 4.1.1. Use the full batch gradient descent method for this smoke test
Method = experiment_cfg["experiment"]["methods"]["full_batch_gd"]

batch_size = Method["batch_size"]
sampling_method = Method["sampling_method"]

data_equivalent_passes = 10 # Or epochs, since we are using the full dataset for each update

learning_rate = 0.01

# Convert the pass budget into the universal accounting variable used by the
# trainer: total examples consumed, independent of batch size.
target_examples_processed = data_equivalent_passes * X_train.shape[0]  # Total number of examples to process during training


print(f"Training parameters: Method={Method['method_name']}, batch_size={batch_size}, sampling_method={sampling_method}, learning_rate={learning_rate}, target_examples_processed={target_examples_processed}")


# 4.1.2. Train using the train_model function

# Reset the model to its initial state before training
model.load_state_dict(initial_state)

# Call the SGD optimiser with the specified learning rate
optimiser = torch.optim.SGD(model.parameters(), lr=float(learning_rate))

# train_model returns a list of checkpoint dictionaries: the initial model
# state plus one row each time the examples-processed checkpoint is reached.

history = train_model(
                model=model,
                optimiser=optimiser,
                loss_function=torch.nn.functional.mse_loss,
                training_data=(X_train, y_train),
                evaluation_data=(X_val, y_val, f_val_true),
                method=Method["method_name"],
                sampling_method=sampling_method,
                batch_size=batch_size,
                target_examples_processed=target_examples_processed,
                sampling_seed=1000, # Sampling seed for reproducibility
                # Checkpoint cadence: pause and evaluate after this many
                # training examples have been consumed. Since n_train = 5000,
                # this gives one evaluation per data-equivalent pass.
                evaluation_every_examples=5000,
            )

# The history contains checkpoint-level diagnostics, including the initial
# untrained model at history[0] and the final checkpoint at history[-1].
print("History length:", len(history))
print("First checkpoint:")
print(history[0])

print("Final checkpoint:")
print(history[-1])

summary_keys = [
    "step",
    "cumulative_examples_processed",
    "data_equivalent_passes",
    "epoch",
    "training_mse",
    "validation_mse",
    "validation_function_mse",
    "update_gradient_norm",
    "full_gradient_norm",
    "parameter_norm",
    "training_elapsed_seconds",
]

for row in [history[0], history[-1]]:
    print({key: row[key] for key in summary_keys})
