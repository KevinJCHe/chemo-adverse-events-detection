"""
========================================================================
© 2018 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
from functools import partial
from bayes_opt import BayesianOptimization

from scripts.utility import (load_ml_model, load_ensemble_weights, load_predictions, save_predictions)
from scripts.model import (MLModels, RNN, IsotonicCalibrator)

from sklearn.metrics import (accuracy_score, roc_auc_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

torch.manual_seed(0)
np.random.seed(0)

class TrainML:
    """
    Train machine learning models
    Employ model calibration and Bayesian Optimization
    """
    def __init__(self, dataset, output_path, n_jobs=32):
        self.ml = MLModels(n_jobs)
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = dataset
        self.target_types = self.Y_train.columns.tolist()
        self.data_splits = {'Train': (self.X_train, self.Y_train),
                            'Valid': (self.X_valid, self.Y_valid),
                            'Test': (self.X_test, self.Y_test)}
        self.output_path = output_path
        self.preds = {split: {algorithm: None for algorithm in self.ml.models} for split in self.data_splits}
        
    def predict(self, model, split, algorithm):
        X, Y = self.data_splits[split]
        if self.preds[split][algorithm] is None:
            pred = model.predict_proba(X)
            # format it to be row=chemo_sessions, columns=targets
            # [:, :, 1] - first column is prob of false, second column is prob of true
            if algorithm != 'NN': pred = np.array(pred)[:, :, 1].T
            # make your life easier by ensuring pred and Y have same data format
            pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
            self.preds[split][algorithm] = pred
        else:
            pred = self.preds[split][algorithm]
        return pred, Y

    def bo_evaluate(self, algorithm, split='Valid', **kwargs):
        """Evaluation function for bayesian optimization
        """
        get_model = getattr(self.ml, f'get_{algorithm}_model')
        kwargs = self.convert_param_types(kwargs)
        model = get_model(**kwargs)
        model.fit(self.X_train, self.Y_train.astype(int)) # astype int because XGB throws a fit if you don't do it
        pred_prob, Y = self.predict(model, split, algorithm)
        self.preds[split][algorithm] = None # reset the cache
        return roc_auc_score(Y, pred_prob) # mean (macro-mean) of auroc scores of all target types
    
    def convert_param_types(self, best_param):
        for param in ['max_depth', 'batch_size', 'n_estimators',
                      'first_layer_size', 'second_layer_size']:
            if param in best_param:
                best_param[param] = int(best_param[param])
        return best_param

    def bayesopt(self, algorithm, random_state=42):
        """Conduct bayesian optimization
        """
        optim_config, hyperparam_config = self.ml.model_tuning_config[algorithm]
        evaluate_function = partial(self.bo_evaluate, algorithm=algorithm)
        bo = BayesianOptimization(evaluate_function, hyperparam_config, random_state=random_state)
        bo.maximize(acq='ei', **optim_config)
        best_param = bo.max['params']
        best_param = self.convert_param_types(best_param)
        logging.info(f'Finished finding best hyperparameters for {algorithm}')
        logging.info(f'Best param: {best_param}')

        # Save the best hyperparameters
        param_filename = f'{self.output_path}/best_params/{algorithm}_classifier_best_param.pkl'
        with open(param_filename, 'wb') as file:    
            pickle.dump(best_param, file)

        return best_param
    
    def train_model_with_best_param(self, algorithm, model, best_param):
        get_model = getattr(self.ml, f'get_{algorithm}_model')
        model = get_model(**best_param)
        model.fit(self.X_train, self.Y_train)

        # Save the model
        model_filename = f'{self.output_path}/{algorithm}_classifier.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

        return model

    def tune_and_train(self, run_bayesopt=True, run_training=True, save_preds=True, skip_alg=None, random_state=42):
        """
        Args:
            skip_alg: list of algorithms you do not want to train/tune
        """
        if skip_alg is None: skip_alg = []
        for algorithm, model in self.ml.models.items():
            if algorithm in skip_alg: continue

            if run_bayesopt:
                best_param = self.bayesopt(algorithm)
            else:
                filename = f'{self.output_path}/best_params/{algorithm}_classifier_best_param.pkl'
                if not os.path.exists(filename): 
                    raise ValueError(f'Please run bayesian optimization for {algorithm} to obtain best hyperparameters')
                with open(filename, 'rb') as file: 
                    best_param = pickle.load(file)

            if run_training:
                if algorithm == 'NN': 
                    best_param['max_iter'] = 100
                    best_param['verbose'] = True
                elif algorithm == 'LR': 
                    best_param['max_iter'] = 1000
                self.train_model_with_best_param(algorithm, model, best_param)
                logging.info(f'{algorithm} training completed!')

        # Get predictions
        for algorithm in self.ml.models:
            model = load_ml_model(self.output_path, algorithm)
            for split in self.data_splits:
                self.predict(model, split, algorithm)
            if save_preds: save_predictions(self.preds, save_dir=f'{self.output_path}/predictions')
    
class SeqData(TensorDataset):
    def __init__(self, mapping, ids):
        self.mapping = mapping
        self.ids = ids
                
    def __getitem__(self, index):
        sample = self.ids[index]
        X, Y = self.mapping[sample]
        features_tensor = torch.Tensor(X.values)
        target_tensor = torch.Tensor(Y.values)
        indices_tensor = torch.Tensor(Y.index)
        return features_tensor, target_tensor, indices_tensor
    
    def __len__(self):
        return(len(self.ids))
    
class TrainRNN(TrainML):
    def __init__(self, dataset, output_path):
        super().__init__(dataset, output_path)
        self.optim_config = {'init_points': 3, 'n_iter': 70}
        self.hyperparam_config = {'batch_size': (8, 512),
                                  'learning_rate': (0.0001, 0.01),
                                  'hidden_size': (10, 200),
                                  'hidden_layers': (1, 5),
                                  'dropout': (0.0, 0.9),
                                  'model': (0.0, 1.0)}
        self.n_features = self.X_train.shape[1] - 1 # -1 for ikn
        self.n_targets = self.Y_train.shape[1]
        self.train_dataset = self.transform_to_tensor_dataset(self.X_train, self.Y_train)
        self.valid_dataset = self.transform_to_tensor_dataset(self.X_valid, self.Y_valid)
        self.test_dataset = self.transform_to_tensor_dataset(self.X_test, self.Y_test)
        self.dataset_splits = {'Train': self.train_dataset, 'Valid': self.valid_dataset, 'Test': self.test_dataset}
        self.preds = {split: pd.DataFrame(index=Y.index) for split, (X, Y) in self.data_splits.items()}
        self.pad_value = -999 # the padding value for padding variable length sequences
        self.calibrator = IsotonicCalibrator(self.target_types)
        
    def transform_to_tensor_dataset(self, X, Y):
        X = X.astype(float)
        mapping = {}
        for ikn, group in X.groupby('ikn'):
            group = group.drop(columns=['ikn'])
            mapping[ikn] = (group, Y.loc[group.index])
            
        return SeqData(mapping=mapping, ids=X['ikn'].unique())

    def get_model(self, load_saved_weights=False, **model_param):
        model = RNN(n_features=self.n_features, n_targets=self.n_targets, pad_value=self.pad_value, **model_param)
        if torch.cuda.is_available():
            model.cuda()
        if load_saved_weights:
            save_path = f'{self.output_path}/{model.name}_classifier'
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            model.load_state_dict(torch.load(save_path, map_location=map_location))
        return model
    
    def format_sequences(self, inputs, targets):
        """
        1. Pad the variable length sequences
        2. Pack the padded sequences to optimize computations 
           (if one of the sequence is significantly longer than other sequences, and we pad all the sequences to the same length, 
            we will be doing a lot of unnecessary computations with the padded values)

        Code Example:
        a = [torch.Tensor([1,2,3]), torch.Tensor([3,4])]
        b = pad_sequence(a, batch_first=True, padding_value=-1)
        >>> tensor([[1,2,3],
                    [3,4,-1]])
        c = pack_padded_sequence(b, batch_first=True, lengths=[3,2])
        >>> PacekedSequence(data=tensor([1,3,2,4,3]), batch_sizes=tensor([2,2,1]))

        data = all the tensors concatenated along the "time" axis. 
        batch_sizes = array of batch sizes at each "time" step. So [2,2,1] represent the grouping [1,3], [2,4], [3]
        """
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_value)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_value)
        
        seq_lengths = list(map(len, inputs))
        packed_padded_inputs = pack_padded_sequence(padded_inputs, seq_lengths, batch_first=True, enforce_sorted=False).float()
        return packed_padded_inputs, padded_targets
    
    def validate(self, model, loader, criterion):
        """
        Refer to train_classification for detailed comments
        """
        total_loss = 0
        total_score = 0
        for i, batch in enumerate(loader):
            inputs, targets, _ = tuple(zip(*batch))
            packed_padded_inputs, padded_targets = self.format_sequences(inputs, targets)
            targets = torch.cat(targets).float()
            if torch.cuda.is_available():
                packed_padded_inputs = packed_padded_inputs.cuda()
                targets = targets.cuda()
            preds = model(packed_padded_inputs)
            preds = preds[padded_targets != self.pad_value].reshape(-1, self.n_targets)
            loss = criterion(preds, targets)
            loss = loss.mean(axis=0)
            total_loss += loss
            preds = torch.sigmoid(preds)
            preds = preds > 0.5
            if torch.cuda.is_available():
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
            total_score += np.array([accuracy_score(targets[:, i], preds[:, i]) for i in range(self.n_targets)])
        return total_loss.cpu().detach().numpy() / (i+1), total_score/(i+1)
    
    def train_model(self, epochs=200, batch_size=512, learning_rate=0.001, decay=0,
                    hidden_size=20, hidden_layers=3, dropout=0.5, model='GRU', 
                    early_stopping=30, pred_threshold=0.5, save=False):
        
        model = self.get_model(load_saved_weights=False, model=model, batch_size=batch_size,
                               hidden_size=hidden_size, hidden_layers=hidden_layers, dropout=dropout)

        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x:x)
        valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x:x)

        best_val_loss = np.inf
        best_model_param = None
        torch.manual_seed(42)

        """
        This loss criterion COMBINES Sigmoid and BCELoss. 
        This is more numerically stable than using Sigmoid followed by BCELoss.
        AS a result, the model does not use a Simgoid layer at the end. 
        The model prediction output will not be bounded from (0, 1).
        In order to bound the model prediction, you must Sigmoid the model output.
        """
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

        train_losses = np.zeros((epochs, self.n_targets))
        valid_losses = np.zeros((epochs, self.n_targets))
        train_scores = np.zeros((epochs, self.n_targets)) # acc score
        valid_scores = np.zeros((epochs, self.n_targets)) # acc score
        counter = 0 # for early stopping

        for epoch in range(epochs):
            train_loss = 0
            train_score = 0
            for i, batch in enumerate(train_loader):
                inputs, targets, _ = tuple(zip(*batch)) # each is a tuple of tensors
                
                # format sequences
                packed_padded_inputs, padded_targets = self.format_sequences(inputs, targets)
                targets = torch.cat(targets).float() # concatenate the tensors
                if torch.cuda.is_available():
                    packed_padded_inputs = packed_padded_inputs.cuda()
                    targets = targets.cuda()

                # Make predictions
                # for each patient, for each timestep, a prediction was made given the prev sequence history of that time step
                preds = model(packed_padded_inputs) 
                preds = preds[padded_targets != self.pad_value] # unpad predictions based on target lengths
                preds = preds.reshape(-1, self.n_targets) # ensure preds shape matches targets shape

                # Calculate loss
                loss = criterion(preds, targets)
                loss = loss.mean(axis=0)
                train_loss += loss
                
                # Bound the model prediction
                preds = torch.sigmoid(preds)

                preds = preds > pred_threshold
                if torch.cuda.is_available():
                    preds = preds.cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                train_score += np.array([accuracy_score(targets[:, i], preds[:, i]) for i in range(self.n_targets)])
                
                loss = loss.mean()
                loss.backward() # back propagation, compute gradients
                optimizer.step() # apply gradients
                optimizer.zero_grad() # clear gradients for next train

            train_losses[epoch] = train_loss.cpu().detach().numpy()/(i+1)
            train_scores[epoch] = train_score/(i+1)
            valid_losses[epoch], valid_scores[epoch] = self.validate(model, valid_loader, criterion)
            statement = f"Epoch: {epoch+1}, " +\
                        f"Train Loss: {train_losses[epoch].mean().round(4)}, " +\
                        f"Valid Loss: {valid_losses[epoch].mean().round(4)}, " +\
                        f"Train Accuracy: {train_scores[epoch].mean().round(4)}, " +\
                        f"Valid Accuracy: {valid_scores[epoch].mean().round(4)}"
            logging.info(statement)

            if valid_losses[epoch].mean() < best_val_loss:
                logging.info('Saving Best Model')
                best_val_loss = valid_losses[epoch].mean()
                best_model_param = model.state_dict()
                counter = 0

            # early stopping
            if counter > early_stopping: 
                train_losses = train_losses[:epoch+1]
                valid_losses = valid_losses[:epoch+1]
                train_scores = train_scores[:epoch+1]
                valid_scores = valid_scores[:epoch+1]
                break
            counter += 1

        if save:
            save_path = f"{self.output_path}/figures/rnn_train_performance"
            np.save(f"{save_path}/train_losses.npy", train_losses)
            np.save(f"{save_path}/valid_losses.npy", valid_losses)
            np.save(f"{save_path}/train_scores.npy", train_scores)
            np.save(f"{save_path}/valid_scores.npy", valid_scores)
            
            save_path = f'{self.output_path}/{model.name}_classifier'
            logging.info(f'Writing best model to {save_path}')
            torch.save(best_model_param, save_path)

        return model
        
    def _get_model_predictions(self, model, split, bound_pred=True):
        """
        Args: 
            bound_pred (bool): bound the predictions by using Sigmoid over the model output
        """
        dataset = self.dataset_splits[split]
        loader = DataLoader(dataset=dataset, batch_size=10000, shuffle=False, collate_fn=lambda x:x)
        pred_arr = np.empty([0, self.n_targets])
        index_arr = np.empty(0)
        for i, batch in enumerate(loader):
            inputs, targets, indices = tuple(zip(*batch))

            packed_padded_inputs, padded_targets = self.format_sequences(inputs, targets)
            indices = torch.cat(indices).float()
            if torch.cuda.is_available():
                packed_padded_inputs = packed_padded_inputs.cuda()

            with torch.no_grad():
                preds = model(packed_padded_inputs)
            preds = preds[padded_targets != self.pad_value].reshape(-1, self.n_targets)
            if bound_pred: 
                preds = torch.sigmoid(preds)

            if torch.cuda.is_available():
                preds = preds.cpu().detach().numpy()

            pred_arr = np.concatenate([pred_arr, preds])
            index_arr = np.concatenate([index_arr, indices])

        # garbage collection
        del preds, targets, packed_padded_inputs
        torch.cuda.empty_cache()
        
        return pred_arr, index_arr
        
    def get_model_predictions(self, model, split, calibrated=False, **kwargs):
        if self.preds[split].empty:
            pred_arr, index_arr = self._get_model_predictions(model, split, **kwargs)
            self.preds[split].loc[index_arr, self.target_types] = pred_arr
            if calibrated: # get the calibrated predictions
                self.preds[split] = self.calibrator.predict(self.preds[split])

        pred = self.preds[split]
        _, target = self.data_splits[split]
        
        return pred, target
    
    def convert_param_types(self, best_param):
        for param in ['batch_size', 'hidden_size', 'hidden_layers']:
            best_param[param] = int(best_param[param])
        best_param['model'] = 'LSTM' if best_param['model'] > 0.5 else 'GRU'
        return best_param

    def bo_evaluate(self, split='Valid', **params):
        params['epochs'] = 15
        params = self.convert_param_types(params)
        logging.info(f"Evaluating parameters: {params}")
        model = self.train_model(**params)
        
        # compute total mean auc score for all target types on the Validation split
        pred_arr, index_arr = self._get_model_predictions(model, split)
        _, target = self.data_splits[split]
        target = target.loc[index_arr]
        auc_score = [roc_auc_score(target[target_type], pred_arr[:, i]) 
                     for i, target_type in enumerate(self.target_types)]
        return np.mean(auc_score)
    
    def bayesopt(self, algorithm='RNN', random_state=42):
        # Conduct Bayesian Optimization
        bo = BayesianOptimization(self.bo_evaluate, self.hyperparam_config, random_state=random_state)
        bo.maximize(acq='ei', **self.optim_config)
        best_param = bo.max['params']
        best_param = self.convert_param_types(best_param)
        logging.info(f'Best param: {best_param}')

        # Save the best hyperparameters
        param_filename = f'{self.output_path}/best_params/{algorithm}_classifier_best_param.pkl'
        with open(param_filename, 'wb') as file:    
            pickle.dump(best_param, file)

        return best_param
    
    def tune_and_train(self, algorithm='RNN', run_bayesopt=True, run_training=True, run_calibration=True, save_preds=True):
        if run_bayesopt:
            best_param = self.bayesopt()
        else:
            filename = f'{self.output_path}/best_params/{algorithm}_classifier_best_param.pkl'
            if not os.path.exists(filename): 
                raise ValueError(f'Please run bayesian optimization for {algorithm} to obtain best hyperparameters')
            with open(filename, 'rb') as file: 
                best_param = pickle.load(file)

        if run_training:
            model = self.train_model(save=True, **best_param)
        else:
            del best_param['learning_rate']
            model = self.get_model(load_saved_weights=True, **best_param)
        
        if run_calibration:
            pred = pd.DataFrame(*self._get_model_predictions(model, 'Valid'), columns=self.target_types)
            self.calibrator.calibrate(pred, self.Y_valid)
            self.calibrator.save_model(self.output_path, algorithm)
        else:
            self.calibrator.load_model(self.output_path, algorithm)
            
        # Get predictions
        for split in self.data_splits:
            self.get_model_predictions(model, split, calibrated=True)
        if save_preds: 
            save_predictions(self.preds, save_dir=f'{self.output_path}/predictions', filename='rnn_predictions')
            
class TrainENS:
    """
    "Train" the ensemble model 
    Find optimal weights via bayesopt
    """
    def __init__(self, output_path, preds, labels):
        """
        Args:
            preds (dict): includes predictions for each split from all models that will be part of the ensemble model
                          e.g. {'Valid': {'LR': pred, 'XGB': pred, 'GRU': pred}
                                'Test': {'LR': pred, 'XGB': pred, 'GRU': pred}}
                                
                                where pred is a pd.DataFrame
                                
            labels (dict): includes labels for each split
                           e.g. {'Valid': label
                                 'Test': label}
                                 
                           where label is a pd.DataFrame
        """
        self.output_path = output_path
        self.preds = preds
        self.labels = labels

        self.splits = list(labels.keys()) 
        self.models = list(preds[self.splits[0]].keys())
        # make sure each split contains the same models
        assert all(list(preds[split].keys()) == self.models for split in self.splits)
        self.target_types = labels[self.splits[0]].columns.tolist()
        
        self.optim_config = {'init_points': 4, 'n_iter': 30}
        self.hyperparam_config = {alg: (0, 1) for alg in self.models}
        
        self.calibrator = IsotonicCalibrator(self.target_types)
        
    def bo_evaluate(self, split='Valid', **kwargs):
        ensemble_weights = [kwargs[algorithm] for algorithm in self.models]
        pred_prob, Y = self.predict(split, ensemble_weights)
        return roc_auc_score(Y, pred_prob) # mean (macro-mean) of auroc scores of all target types
    
    def bayesopt(self, algorithm='ENS', random_state=42):
        # Conduct Bayesian Optimization
        bo = BayesianOptimization(self.bo_evaluate, self.hyperparam_config, random_state=random_state)
        bo.maximize(acq='ei', **self.optim_config)
        best_param = bo.max['params']
        logging.info(f'Best param: {best_param}')

        # Save the best hyperparameters
        param_filename = f'{self.output_path}/best_params/{algorithm}_classifier_best_param.pkl'
        with open(param_filename, 'wb') as file:    
            pickle.dump(best_param, file)

        return best_param
    
    def predict(self, split, ensemble_weights=None):
        # compute ensemble predictions by soft vote
        if ensemble_weights is None: ensemble_weights = [1,]*len(self.models)
        pred = [self.preds[split][algorithm] for algorithm in self.models]
        pred = np.average(pred, axis=0, weights=ensemble_weights)
        Y = self.labels[split]
        pred = pd.DataFrame(pred, index=Y.index, columns=Y.columns)
        return pred, Y
    
    def store_prediction(self, calibrated=False):
        ensemble_weights = load_ensemble_weights(f'{self.output_path}/best_params')
        ensemble_weights = [ensemble_weights[algorithm] for algorithm in self.models]
        for split in self.splits:
            pred_prob, _ = self.predict(split, ensemble_weights)
            if calibrated:
                pred_prob = self.calibrator.predict(pred_prob)
            self.preds[split]['ENS'] = pred_prob
            
    def tune_and_train(self, algorithm='ENS', run_bayesopt=True, run_calibration=True):
        if run_bayesopt:
            best_param = self.bayesopt()
        else:
            filename = f'{self.output_path}/best_params/{algorithm}_classifier_best_param.pkl'
            if not os.path.exists(filename): 
                raise ValueError(f'Please run bayesian optimization for {algorithm} to obtain best hyperparameters')
            with open(filename, 'rb') as file: 
                best_param = pickle.load(file)
        ensemble_weights = [best_param[algorithm] for algorithm in self.models]
        
        if run_calibration:
            self.calibrator.calibrate(*self.predict('Valid', ensemble_weights))
            self.calibrator.save_model(self.output_path, algorithm)
        else:
            self.calibrator.load_model(self.output_path, algorithm)
            
        # Store predictions
        self.store_prediction(calibrated=True)
            
