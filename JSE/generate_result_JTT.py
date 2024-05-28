


import numpy as np
import pandas as pd
import torch


from JSE.data import *
from JSE.settings import data_info, optimizer_info
from JSE.models import *
from JSE.training import *

import argparse
import os
import sys

# import logistic regression from sklearn
from sklearn.linear_model import LogisticRegression


def main(dataset, dataset_setting, spurious_ratio, demean, pca, k_components, weight_misclassified, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs, balanced_training_main,  per_step, device_type, save_results, seed, folder, use_standard_ERM_for_model_after=True, single_finetuned_model=False):

    # set the device
    device = torch.device(device_type)


    # save all the results in a dict for this run
    results_dict = { 'method': 'JTT', 'parameters' :{'dataset': dataset,'spurious_ratio': spurious_ratio,  'demean': demean, 'pca': pca, 'k_components': k_components, 'weight_misclassified': weight_misclassified, 'alpha': alpha, 'batch_size': batch_size, 'solver': solver, 'lr': lr, 'weight_decay': weight_decay, 'early_stopping': early_stopping, 'epochs': epochs, 'balanced_training_main': balanced_training_main, 'per_step': per_step, 'device_type': device_type,  'seed': seed}}


    # set the settings for dataset
    dataset_settings = data_info[dataset][dataset_setting]
    optimizer_settings = optimizer_info['All']


    # get the data obj
    set_seed(seed)
    data_obj = get_dataset_obj(dataset, dataset_settings, spurious_ratio, data_info, seed, device, use_punctuation_MNLI=True, single_model_for_embeddings=single_finetuned_model   )

    # demean, pca
    if demean:
        data_obj.demean_X(reset_mean=True, include_test=True)
    if pca:
        data_obj.transform_data_to_k_components(k_components, reset_V_k=True, include_test=True)
        V_k_train = data_obj.V_k_train

    # get the data
    X_train, y_c_train, y_m_train = data_obj.X_train, data_obj.y_c_train, data_obj.y_m_train
    X_val, y_c_val, y_m_val = data_obj.X_val, data_obj.y_c_val, data_obj.y_m_val
    X_test, y_c_test, y_m_test = data_obj.X_test, data_obj.y_c_test, data_obj.y_m_test

    # get the main task weights
    main_weights_train, main_weights_val = None, None
    include_weights = False

    # if class-balanced training for the main task 
    if balanced_training_main:
        main_weights_train, main_weights_val = data_obj.get_class_weights_train_val(y_m_train, y_m_val)
        include_weights = True
        print('used balanced weights')
    
     
    
    # get the optimizer settings
    if use_standard_ERM_for_model_after:
        

        lr_after = optimizer_info['standard_ERM_settings'][dataset]['lr']
        weight_decay_after = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
        batch_size_after = optimizer_info['standard_ERM_settings'][dataset]['batch_size']
    
    else:
        lr_after = lr
        weight_decay_after = weight_decay
        batch_size_after = batch_size

        
    lr_identifier = lr
    weight_decay_identifier = weight_decay
    batch_size_identifier = batch_size


    # reset the data objects
    data_obj.reset_X(X_train, X_val, batch_size=batch_size_identifier, reset_X_objects=False, include_weights=include_weights, train_weights = main_weights_train, val_weights = main_weights_val)
    d = X_train.shape[1]

    # Train the model to identify the spurious features
    set_seed(seed)
    identifier_model = return_linear_model(d, 
                                               data_obj.main_loader,
                                              device,
                                              solver = solver,
                                              lr=lr_identifier,
                                              per_step=per_step,
                                              tol = optimizer_settings['tol'],
                                              early_stopping = early_stopping,
                                              patience = optimizer_settings['patience'],
                                              epochs = epochs,
                                              bias=True,
                                              weight_decay=weight_decay_identifier, 
                                              model_name=dataset+'_main_model')
    
    # get predictions of the identifier model
    y_m_pred_train = identifier_model(X_train)
    y_m_pred_val = identifier_model(X_val)

    # get the JTT weights
    JTT_weights_train = get_JTT_weights(y_m_train, y_m_pred_train, weight_misclassified=weight_misclassified)
    JTT_weights_val = get_JTT_weights(y_m_val, y_m_pred_val, weight_misclassified=weight_misclassified)
    
    # reset the data objects
    print('JTT_weights_train', JTT_weights_train)
    data_obj.reset_X(X_train, X_val, batch_size=batch_size_after, reset_X_objects=True, include_weights=include_weights, train_weights = main_weights_train, val_weights = main_weights_val, add_indeces=True)
    d = X_train.shape[1]


    # Train the model to get JTT
    JTT_model = return_linear_model(d,
                                    data_obj.main_loader,
                                    device,
                                    solver = solver,
                                    lr=lr_after,
                                    per_step=per_step,
                                    tol = optimizer_settings['tol'],
                                    early_stopping = early_stopping,
                                    patience = optimizer_settings['patience'],
                                    epochs = epochs,
                                    bias=True,
                                    weight_decay=weight_decay_after, 
                                    model_name=dataset+'_JTT_model',
                                    weights_BCE_train=JTT_weights_train,
                                    weights_BCE_val=JTT_weights_val,
                                    )


    y_m_pred_train = JTT_model(X_train)
    y_m_pred_val = JTT_model(X_val)
    y_m_pred_test = JTT_model(X_test)


    # get the accuracy of the main model overall 
    main_acc_after = get_acc_pytorch_model(y_m_test, y_m_pred_test)
    main_acc_train_after = get_acc_pytorch_model(y_m_train, y_m_pred_train)
    main_acc_val_after = get_acc_pytorch_model(y_m_val, y_m_pred_val)


    # get the accuracy of the main model per group
    result_per_group, _ = get_acc_per_group(y_m_pred_test, y_m_test, y_c_test)
    result_per_group_train, _ = get_acc_per_group(y_m_pred_train, y_m_train, y_c_train)
    result_per_group_val, _ = get_acc_per_group(y_m_pred_val, y_m_val, y_c_val)
    BCE_main = torch_calc_BCE(y_m_test,y_m_pred_test, device)

    print("Overall Accuracy of JTT (test): ", main_acc_after)
    print("Overall Accuracy of JTT (train): ", main_acc_train_after)
    print("Overall Accuracy of JTT (val): ", main_acc_val_after)
    print("Accuracy per group of JTT (test): ", result_per_group)
    print("Accuracy per group of JTT (train): ", result_per_group_train)
    print("Accuracy per group of JTT (val): ", result_per_group_val)

    print("BCE (main) on the test set: ", BCE_main )


    if save_results:
        results_dict['overall_acc_test'] = main_acc_after  
        results_dict['result_per_group_test'] = result_per_group
        results_dict['result_per_group_train'] = result_per_group_train
        results_dict['result_per_group_val'] = result_per_group_val
        results_dict['weights'] = JTT_model.linear_layer.weight.data.detach()
        results_dict['b'] = JTT_model.linear_layer.bias.data.detach()
        results_dict['param_W'] = JTT_model.state_dict()
        results_dict['V_c'] = None
        results_dict['X_train'] = X_train
        results_dict['X_val'] = X_val
        results_dict['X_test'] = X_test
        results_dict['y_c_train'] = y_c_train
        results_dict['y_c_val'] = y_c_val
        results_dict['y_c_test'] = y_c_test
        results_dict['y_m_train'] = y_m_train
        results_dict['y_m_val'] = y_m_val
        results_dict['y_m_test'] = y_m_test
        if pca:
            results_dict['V_k_train'] = V_k_train
        if demean:
            results_dict['X_train_mean'] = data_obj.X_train_mean

         # Check whether the specified path exists or not
        folder_exists = os.path.exists('results/'+ folder)
        if not folder_exists:
            # Create a new directory because it does not exist
            os.makedirs('results/'+ folder)

        

        # save the results_dict in results folder
        filename = 'results/{}/{}_{}_seed_{}'.format(folder, dataset, 'JTT', seed)
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(results_dict, fp)
            print('dictionary saved successfully to file')
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
    parser.add_argument("--spurious_ratio", type=float, default=0.5, help="ratio of spurious features")
    parser.add_argument("--dataset_setting", type=str, default="default", help="dataset setting to use")
    parser.add_argument("--demean", type=str,  help="whether to demean the data")
    parser.add_argument("--pca", type=str, default=False, help="whether to use pca")
    parser.add_argument("--k_components", type=int, default=10, help="number of components to use for pca")
    parser.add_argument("--alpha", type=float, default=0.05, help="alpha to use for hypothesis testing")
    parser.add_argument("--weight_misclassified", type=float,  help="whether to weight misclassified samples more")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size to use")
    parser.add_argument("--solver", type=str, default="SGD", help="solver to use")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate to use")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay to use")
    parser.add_argument("--early_stopping", type=str, help="whether to use early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to use")
    parser.add_argument("--balanced_training_main", type=str, help="whether to use balanced training for main")
    parser.add_argument("--per_step", type=int, default=1, help="per steps, print the loss")
    parser.add_argument("--device_type", type=str, default="cuda", help="device to use")
    parser.add_argument("--save_results",type=str )
    parser.add_argument("--seed", type=int, default=0, help="seed to use")
    parser.add_argument("--folder", type=str, default="default", help="folder to save results in")
    parser.add_argument('--use_standard_ERM_settings', type=str, default=True, help="whether to use standard ERM settings")
    parser.add_argument('--single_finetuned_model', type=str, default=False, help="whether to use a single finetuned model for embeddings")

    args = parser.parse_args()
    dict_arguments = vars(args)

    dataset = dict_arguments["dataset"]
    spurious_ratio = dict_arguments["spurious_ratio"]
    dataset_setting = dict_arguments["dataset_setting"]
    demean = str_to_bool(dict_arguments["demean"])
    pca = str_to_bool(dict_arguments["pca"])
    k_components = dict_arguments["k_components"]
    weight_misclassified = (dict_arguments["weight_misclassified"])
    alpha = dict_arguments["alpha"]
    batch_size = dict_arguments["batch_size"]
    solver = dict_arguments["solver"]
    lr = dict_arguments["lr"]
    weight_decay = dict_arguments["weight_decay"]
    early_stopping = str_to_bool(dict_arguments["early_stopping"])
    epochs = dict_arguments["epochs"]
    balanced_training_main = str_to_bool(dict_arguments["balanced_training_main"])
    per_step = dict_arguments["per_step"]
    device_type = dict_arguments["device_type"]
    save_results = dict_arguments["save_results"]
    seed = dict_arguments["seed"]
    folder = dict_arguments["folder"]
    single_finetuned_model = str_to_bool(dict_arguments["single_finetuned_model"])

    main(dataset, dataset_setting, spurious_ratio, demean, pca, k_components, weight_misclassified, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs, balanced_training_main,  per_step, device_type, save_results, seed, folder, single_finetuned_model=single_finetuned_model)


