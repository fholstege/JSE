


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


def main(dataset, dataset_setting, spurious_ratio, demean, pca, k_components, C, eta_g, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs, balanced_training_main,  per_step, device_type, save_results, seed, folder, use_smaller_set=False, n_for_estimation='all', single_finetuned_model=False):

    # set the device
    device = torch.device(device_type)



    # save all the results in a dict for this run
    results_dict = { 'method': 'GDRO', 'parameters' :{'dataset': dataset,'spurious_ratio': spurious_ratio,  'demean': demean, 'pca': pca, 'k_components': k_components, 'C': C, 'eta_g':eta_g, 'alpha': alpha, 'batch_size': batch_size, 'solver': solver, 'lr': lr, 'weight_decay': weight_decay, 'early_stopping': early_stopping, 'epochs': epochs, 'balanced_training_main': balanced_training_main, 'per_step': per_step, 'device_type': device_type,  'seed': seed}}


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
    d = X_train.shape[1]


    # get the main task weights
   
    main_weights_train, main_weights_val = None, None

    # get the data
    if balanced_training_main:
        main_weights_train, main_weights_val = data_obj.get_class_weights_train_val(y_m_train, y_m_val)
        print('balanced training main')

    else:
        main_weights_train, main_weights_val = None, None


    start_model_loader = data_obj.create_loaders( batch_size, 0, shuffle=True, with_concept=False, which_dependent='main', include_weights=balanced_training_main, train_weights = main_weights_train, val_weights = main_weights_val)

    # Train the model to get main
    print("Set starting weights")
    set_seed(seed)
    start_lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
    start_weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']

    start_model = return_linear_model(d,
                                                    start_model_loader,
                                                    device,
                                                    solver = solver,
                                                    lr=start_lr,
                                                    per_step=per_step,
                                                    tol = optimizer_settings['tol'],
                                                    early_stopping = early_stopping,
                                                    patience = optimizer_settings['patience'],
                                                    epochs = epochs,
                                                    bias=True,
                                                    weight_decay=start_weight_decay, 
                                                    model_name=dataset+'_start_model1',
                                                    GDRO=False,
                                                    )
        

    # if use_smaller_set; use the val set as as training set, split it in 80/20 for training and validation
    if use_smaller_set:

        if n_for_estimation == 'all':
            n_eval = X_val.shape[0]
        else:
            n_eval = int(n_for_estimation)
            
        
        # get the ids for the split
        p_m_subsample =0.5
        n_train = int(0.8*n_eval)
        n_val = n_eval - n_train
        ids_sample_train = data_obj.set_sample_split( y_m_train, y_c_train, n_train, spurious_ratio, p_m_subsample,  seed)
        ids_sample_val = data_obj.set_sample_split( y_m_val, y_c_val, n_val, spurious_ratio, p_m_subsample,  seed)

        # get the data - train
        X_train_new = X_train[ids_sample_train, :]
        y_c_train_new = y_c_train[ids_sample_train]
        y_m_train_new = y_m_train[ids_sample_train]

        # get the data - val
        X_val_new = X_val[ids_sample_val, :]
        y_c_val_new = y_c_val[ids_sample_val]
        y_m_val_new = y_m_val[ids_sample_val]

        # reset the data
        data_obj.X_train = X_train_new
        data_obj.y_c_train = y_c_train_new
        data_obj.y_m_train = y_m_train_new
        data_obj.X_val = X_val_new
        data_obj.y_c_val = y_c_val_new
        data_obj.y_m_val = y_m_val_new 

        print('n_eval: ', n_eval)
        print('n_train: ', n_train)
        print('n_val: ', n_val)



    


    start_model_state_dict = start_model.state_dict()
        

     
    # reset the data objects
    main_weights_train, main_weights_val = data_obj.get_group_weights()
    GDRO_loaders = data_obj.create_loaders( batch_size, 0, shuffle=True, with_concept=True, which_dependent='main', include_weights=True, train_weights = main_weights_train, val_weights = main_weights_val, concept_first=False, add_indeces=False)
        

    # Train the model to get main
    set_seed(seed)
    GDRO_model = return_linear_model(d, 
                                               GDRO_loaders,
                                              device,
                                              solver = solver,
                                              lr=lr,
                                              per_step=per_step,
                                              tol = optimizer_settings['tol'],
                                              early_stopping = early_stopping,
                                              patience = optimizer_settings['patience'],
                                              epochs = epochs,
                                              bias=True,
                                              weight_decay=weight_decay, 
                                              model_name=dataset+'_GDRO_main1',
                                              GDRO=True,
                                              C=C,
                                              eta_g=eta_g,
                                              init_state_dict=start_model_state_dict)
    
    # get the accuracy of the main model
    y_m_pred_test = GDRO_model(X_test)
    y_m_pred_train = GDRO_model(X_train)
    y_m_pred_val = GDRO_model(X_val)

    # get the accuracy of the main model overall 
    main_acc_after = get_acc_pytorch_model(y_m_test, y_m_pred_test)
    main_acc_train_after = get_acc_pytorch_model(y_m_train, y_m_pred_train)
    main_acc_val_after = get_acc_pytorch_model(y_m_val, y_m_pred_val)


    # get the accuracy of the main model per group
    result_per_group, _ = get_acc_per_group(y_m_pred_test, y_m_test, y_c_test)
    result_per_group_train, _ = get_acc_per_group(y_m_pred_train, y_m_train, y_c_train)
    result_per_group_val, _ = get_acc_per_group(y_m_pred_val, y_m_val, y_c_val)
    BCE_main = torch_calc_BCE(y_m_test,y_m_pred_test, device)

    print("Overall Accuracy of GDRO (test): ", main_acc_after)
    print("Overall Accuracy of GDRO (train): ", main_acc_train_after)
    print("Overall Accuracy of GDRO (val): ", main_acc_val_after)
    print("Accuracy per group of GDRO (test): ", result_per_group)
    print("Accuracy per group of GDRO (train): ", result_per_group_train)
    print("Accuracy per group of GDRO (val): ", result_per_group_val)

    print("BCE (main) on the test set: ", BCE_main )


    if save_results:
        results_dict['overall_acc_test'] = main_acc_after  
        results_dict['result_per_group_test'] = result_per_group
        results_dict['result_per_group_train'] = result_per_group_train
        results_dict['result_per_group_val'] = result_per_group_val
        results_dict['weights'] = GDRO_model.linear_layer.weight.data.detach()
        results_dict['b'] = GDRO_model.linear_layer.bias.data.detach()
        results_dict['param_W'] = GDRO_model.state_dict()
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
        filename = 'results/{}/{}_{}_seed_{}'.format(folder, dataset, 'GDRO', seed)
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
    parser.add_argument("--C", type=float,  help="value of C for GDRO")
    parser.add_argument("--eta_g", type=float,  help="value of eta_g for GDRO")
    parser.add_argument('--use_smaller_set', type=str, default='False', help="whether to use the val set as training set")
    parser.add_argument('--n_for_estimation', type=str, default='all', help="whether to use the val set as training set")
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
    parser.add_argument('--single_finetuned_model', type=str, default=False, help="whether to use a single finetuned model")

    args = parser.parse_args()
    dict_arguments = vars(args)

    dataset = dict_arguments["dataset"]
    spurious_ratio = dict_arguments["spurious_ratio"]
    dataset_setting = dict_arguments["dataset_setting"]
    demean = str_to_bool(dict_arguments["demean"])
    pca = str_to_bool(dict_arguments["pca"])
    k_components = dict_arguments["k_components"]
    C = (dict_arguments["C"])
    eta_g = (dict_arguments["eta_g"])
    print('dict arg use smaller set: ', dict_arguments["use_smaller_set"])
    use_smaller_set = str_to_bool(dict_arguments["use_smaller_set"])
    n_for_estimation = dict_arguments["n_for_estimation"]
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

    main(dataset, dataset_setting, spurious_ratio, demean, pca, k_components, C, eta_g, alpha,  batch_size, solver, lr,weight_decay,  early_stopping, epochs, balanced_training_main,  per_step, device_type, save_results, seed, folder, use_smaller_set=use_smaller_set, n_for_estimation=n_for_estimation, single_finetuned_model=single_finetuned_model)


