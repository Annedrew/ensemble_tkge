import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np

# Visualization of study history
from optuna.visualization import plot_optimization_history
import plotly.io as pio

DEVICE = torch.device("cpu")
BATCHSIZE = 128
INPUT_SIZE = 5
OUTPUT_SIZE = 5
DIR = os.getcwd()
EPOCHS = 50


class DDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)


    def __getitem__(self, index):
        # The weight matrix of nn.Linear using float32, so there must be float32
        input_sample = self.data.iloc[index, :INPUT_SIZE].values.astype(np.float32)
        target_label = self.data.iloc[index, - OUTPUT_SIZE:].values.astype(np.float32)

        input_sample = torch.tensor(input_sample)
        target_label = torch.tensor(target_label)

        return input_sample, target_label
    

    def __len__(self):
        return len(self.data)


class HyperTune:
    # Define the architecture of model, also define the range of hyperparameters need to tune,
    def define_model(self, trial):
        n_layers = trial.suggest_int("n_layers", 1, 6)
        layers = []
        # The input layer has INPUT_SIZE neurons.
        in_features = INPUT_SIZE
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            # activation_candidates = {
            #     "relu": torch.nn.ReLU(),
            #     "sigmoid": torch.nn.Sigmoid(),
            #     "tanh": torch.nn.Tanh(),
            #     "softmax": torch.nn.Softmax(dim=1)
            # }
            # activation_name = trial.suggest_categorical("activation", list(activation_candidates))
            # activation_function = activation_candidates[activation_name]
            # layers.append(activation_function)
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
            layers.append(nn.Dropout(p))
            in_features = out_features
        
        layers.append(nn.Linear(in_features, OUTPUT_SIZE))

        return nn.Sequential(*layers)

    # Get the dataset
    def get_dataset(self, train_path, valid_path):
        train_dataset = DDataset(train_path)
        valid_dataset = DDataset(valid_path)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCHSIZE,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCHSIZE,
            shuffle=True,
        )

        return train_loader, valid_loader


    def objective(self, trial):
        # Generate the model.
        model = self.define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get the FashionMNIST dataset.
        train_loader, valid_loader = self.get_dataset("dataset/NN/5p_5w/dataset/train_dataset.csv", "dataset/NN/5p_5w/dataset/validation_dataset.csv")

        loss_function = nn.MSELoss()
        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for data, target in train_loader:
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                # clear the gradient from previous iteration
                optimizer.zero_grad()
                output = model(data)
                mse = loss_function(output, target)
                rmse = torch.sqrt(mse)
                rmse.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            error = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    mse = loss_function(output, target)
                    rmse = torch.sqrt(mse)
                    
                    error += rmse

            mean_error = error / len(valid_loader.dataset)

            trial.report(mean_error, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return mean_error


if __name__ == "__main__":
    tune = HyperTune()
    study = optuna.create_study(direction="minimize")
    study.optimize(tune.objective, n_trials=100, timeout=7200)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    

    plotly_config = {"staticPlot": True}
    fig = plot_optimization_history(study)
    fig.show(config=plotly_config)
    # pio.write_image(fig, "nn_method/att_study.png")