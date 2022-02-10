from typing import Any, Dict

import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import zero

from copy import deepcopy as deepcopy

import os
import sys

import sys

from lib.deep import IndexLoader
import pandas as pd
import rtdl
from data import data as dta



def apply_model(x_num, x_cat=None, model=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(part, model, X, y, y_std, task_type="regression"):
    model.eval()
    prediction = []

    batch_size = 1024
    permutation = torch.randperm(X[part].size()[0])

    for iteration in range(0, X[part].size()[0], batch_size):

        batch_idx = permutation[iteration:iteration + batch_size]

        x_batch = X[part][batch_idx]

        prediction.append(apply_model(x_batch, model=model))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return score



def learn_that(_model, _optimizer, _loss_fn, _X, _y, _epochs, _batch_size, _gse, _old_x, print_mode=False, _task_type="regression", sparse=False):

    size = _X['train'].size()[0]
    column_count = len(_old_x['train'].columns)
    losses = {"val": [], "test": []}
    for epoch in range(1, _epochs + 1):

        print("epoch " + str(epoch) + " on " + str(_epochs) + " epochs \n")
        permutation = torch.randperm(size)

        for iteration in range(0, size, _batch_size):

            batch_idx = permutation[iteration:iteration + _batch_size]

            _model.train()
            _optimizer.zero_grad()
            x_batch = _X['train'][batch_idx]
            y_batch = _y['train'][batch_idx]

            ypred_batch = apply_model(x_batch, model=_model).squeeze(1)

            loss = _loss_fn(ypred_batch, y_batch)
            loss.backward()

            _old_train_x_data = _old_x['train'][iteration * _batch_size:(iteration+1) * _batch_size]

            # Modify gradients
            if _gse:
                old_params = []
                for name, param in _model.named_parameters():
                    if (name == "blocks.0.linear.weight") or (name=="first_layer.weight"):
                        factors = torch.ones(column_count, param.grad.shape[0])
                        for i in range(column_count):
                            idx = _old_train_x_data.columns[i]
                            real_count = _old_train_x_data[idx].sum()
                            if real_count > 0:
                                factors[i] = (_batch_size / (1.0 * real_count)) * factors[i]
                        param.grad = torch.mul(param.grad, torch.transpose(factors,0,1))
                        old_params.append(deepcopy(param))



            if sparse:
                for p in _model.parameters():
                    p.grad = p.grad.to_sparse()

            _optimizer.step()
            if _gse and not sparse:

                i = 0
                for name, param in _model.named_parameters():
                    if (name == "blocks.0.linear.weight") or (name=="first_layer.weight"):
                        param = torch.where(param.grad == 0, old_params[i], param)
                        i += 1


        def get_accuracy(mode):
            count = _y[mode].shape[0]

            y1 = 1 + torch.zeros(count)
            y0 = torch.zeros(count)
            trigger = 0.5 + torch.zeros(count)
            yval = apply_model(_X[mode],model=_model)
            yval = torch.sigmoid(yval.reshape([count]))
            yval = torch.where(yval > trigger, y1, y0)
            acc = float(torch.sum(yval == _y[mode])) / count

            return acc

        if _task_type == "binclass":
            val_acc = get_accuracy("val")
            test_acc = get_accuracy("test")

            losses['val'].append(1 - val_acc)
            losses['test'].append(1 - test_acc)

        else:
            losses['val'].append(float(_loss_fn(apply_model(_X['val'],   model=_model).squeeze(1), _y['val'])))
            losses['test'].append(float(_loss_fn(apply_model(_X['test'], model=_model).squeeze(1), _y['test'])))
        if print_mode:
            batch = "batch"
            if _gse:
                batch= "gse-batch"
            valoss = losses['val'][-1]
            teloss = losses['test'][-1]
            print(f'(VALIDATION epoch) {epoch} ({batch}) (loss) {valoss}')
            print(f'(TEST epoch) {epoch} ({batch}) (loss) {teloss}')



    return losses
