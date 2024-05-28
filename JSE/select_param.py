


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


def main(method, param_name, sims, dataset, spurious_ratio, dataset_setting, demean, pca, k_components,   batch_size, solver, lr, weight_decay, early_stopping, epochs,  per_step, device_type, start_seed, balanced_training_main=False, balanced_training_concept=False,eval_balanced=True, metric='BCE', group_weighted=False, weight_misclassified=1.0, use_smaller_set=True, n_for_estimation=1000, C=[0], eta_g=[0.1], single_finetuned_model=False, use_second_concept=True):

    # create all the combinations between the different values in weight_decay, lr, batch_size
    combinations = []

    if dataset == 'Toy':
        balanced_training_concept = False
        balanced_training_main = False
    elif dataset == 'celebA':
        balanced_training_concept = False
        balanced_training_main = False

    # check if weight decay is a list, if not make it
    if not isinstance(weight_decay, list):
        weight_decay = [weight_decay]
    if not isinstance(lr, list):
        lr = [lr]
    if not isinstance(batch_size, list):
        batch_size = [batch_size]
    if not isinstance(weight_misclassified, list):
        weight_misclassified = [weight_misclassified]


    for weight_decay_i in weight_decay:
        for lr_i in lr:
            for batch_size_i in batch_size:
               

                if method == 'JTT':
                    for weight_misclassified_i in weight_misclassified:
                        combination_i = {'weight_decay': weight_decay_i, 'lr': lr_i, 'batch_size': batch_size_i, 'weight_misclassified': weight_misclassified_i}
                        combinations.append(combination_i)
                elif method == 'GDRO':
                    for C_i in C:
                        for eta_g_i in eta_g:
                            combination_i = {'weight_decay': weight_decay_i, 'lr': lr_i, 'batch_size': batch_size_i, 'C': C_i, 'eta_g': eta_g_i}
                            combinations.append(combination_i)
                else:
                    combination_i = {'weight_decay': weight_decay_i, 'lr': lr_i, 'batch_size': batch_size_i}
                    combinations.append(combination_i)


    # create a dict with the combinations
    n_combinations = len(combinations)
    combinations_dict = {i: {'param': combinations[i]} for i in range(0, n_combinations)}

    # loop over each param type
    i = 0
    for combination in combinations:
        
        # get the parameters
        weight_decay_i = combination['weight_decay']
        lr_i = combination['lr']
        batch_size_i = combination['batch_size']

        if method == 'JTT':
            weight_misclassified_i = combination['weight_misclassified']

        if method == 'GDRO':
            C_i = combination['C']
            eta_g_i = combination['eta_g']


        print('Checking for combination: weight_decay: {}, lr: {}, batch_size: {}'.format(weight_decay_i, lr_i, batch_size_i))
        if method == 'JTT':
            print('and  weight_misclassified: {}'.format(weight_misclassified_i))
        if method == 'GDRO':
            print('and  C: {}, eta_g: {}'.format(C_i, eta_g_i))
        combinations_dict[i]['results'] = [None]*sims


        # loop over the simulations
        for run_i in range(sims):
                
                # set the seed per run
                seed = start_seed + run_i
                
                # set the device
                device = torch.device(device_type)

                # set the settings for dataset
                dataset_settings = data_info[dataset][dataset_setting]
                optimizer_settings = optimizer_info['All']

            
                # get the data obj
                set_seed(seed)
                if not single_finetuned_model:
                    seed_data = run_i
                else:
                    seed_data = seed
                data_obj = get_dataset_obj(dataset, dataset_settings, spurious_ratio, data_info, seed_data, device, use_punctuation_MNLI=True, single_model_for_embeddings=single_finetuned_model)

                if use_second_concept:
                    data_obj.y_c_train = data_obj.y_c_2_train
                    data_obj.y_c_val = data_obj.y_c_2_val
                    data_obj.y_c_test = data_obj.y_c_2_test
                
                
                # demean and pca
                if pca:
                    data_obj.transform_data_to_k_components(k_components, reset_V_k=True, include_test=True)
                    V_k_train = data_obj.V_k_train

                if demean:
                    data_obj.demean_X(reset_mean=True, include_test=True)


                X_train, y_c_train, y_m_train = data_obj.X_train, data_obj.y_c_train, data_obj.y_m_train
                X_val, y_c_val, y_m_val = data_obj.X_val, data_obj.y_c_val, data_obj.y_m_val

                # if use_smaller_set; use the val set as as training set, split it in 80/20 for training and validation
                if use_smaller_set:

                    if n_for_estimation == 'all':
                        n_eval = X_val.shape[0]
                    else:
                        n_eval = int(n_for_estimation)

                    if method == 'SUBG':
                        p_m_subsample = 0.5
                        spurious_ratio_subsample = 0.5
                        equal_to_smallest=True
                    else:
                        p_m_subsample = y_m_val.float().mean()
                        spurious_ratio_subsample = spurious_ratio
                        equal_to_smallest = False

                    
                    # get the ids for the split
                    n_train = int(0.8*n_eval)
                    n_val = n_eval - n_train
                    ids_sample_train = data_obj.set_sample_split( y_m_train, y_c_train, n_train, spurious_ratio_subsample, p_m_subsample,  seed, equal_to_smallest=equal_to_smallest)
                    ids_sample_val = data_obj.set_sample_split( y_m_val, y_c_val, n_val, spurious_ratio_subsample, p_m_subsample,  seed, equal_to_smallest=equal_to_smallest)

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

                
                if dataset == 'Waterbirds' and dataset_setting == 'default_finetuned':
                    balanced_training_concept = True
                    balanced_training_main = True
               

                # calculate the weights for the training and validation set
                if balanced_training_concept:
                    if use_smaller_set:
                        concept_weights_train, concept_weights_val = data_obj.get_class_weights_train_val(y_c_train_new, y_c_val_new)
                    else:
                        concept_weights_train, concept_weights_val = data_obj.get_class_weights_train_val(y_c_train, y_c_val)
                else:
                    concept_weights_train, concept_weights_val= None, None

                # calculate the weights for the training and validation set
                if balanced_training_main:
                    if use_smaller_set:
                        main_weights_train, main_weights_val = data_obj.get_class_weights_train_val(y_m_train_new, y_m_val_new)
                    else:
                        main_weights_train, main_weights_val= data_obj.get_class_weights_train_val(y_m_train, y_m_val)
                elif group_weighted:
                    main_weights_train, main_weights_val = data_obj.get_group_weights()
                else:
                    main_weights_train, main_weights_val= None, None


                if method == 'JSE':

                    # define the loaders
                    loaders =  data_obj.create_loaders(batch_size=batch_size_i, workers=0, with_concept=True, include_weights=balanced_training_concept, train_weights=concept_weights_train, val_weights=concept_weights_val)


                     # get the model
                    d = data_obj.X_train.shape[1]
                    joint_model = return_joint_main_concept_model(d,  
                                                  loaders, 
                                                  device,
                                                               solver=solver,
                                                               lr=lr_i,
                                                               weight_decay=weight_decay_i,
                                                               per_step=per_step,
                                                               tol=optimizer_settings['tol'],
                                                               early_stopping=early_stopping,
                                                               patience=optimizer_settings['patience'],
                                                               epochs=epochs,
                                                               bias=True,
                                                               model_name = 'select_param_JSE_'+dataset,
                                                               save_best_model=True
                                                               )
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    output = joint_model(X_val)
                    y_c_val_pred = output['y_c_1']
                    y_m_val_pred = output['y_m_1']



                    # calculate the binary cross-entropy of the models
                    if metric == 'WG':
                        accuracy_per_group, _ =  get_acc_per_group(y_m_val_pred, y_m_val, y_c_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy
                        print('Worst group accuracy: ', worst_group_accuracy)
                    else:
                        if eval_balanced:
                                BCE_concept = torch_calc_BCE_weighted(
                                            y_m_val, y_c_val, y_c_val_pred, device, reduce='mean', type_dependent='concept')
                                BCE_main = torch_calc_BCE_weighted(
                                            y_m_val, y_c_val, y_m_val_pred, device, reduce='mean', type_dependent='main')
                        else:
                                BCE_concept = torch_calc_BCE(y_c_val,y_c_val_pred, device, reduce='mean')
                                BCE_main = torch_calc_BCE(y_m_val,y_m_val_pred, device, reduce='mean')

                        print('BCE concept: ', BCE_concept.detach().item())
                        print('BCE main: ', BCE_main.detach().item())
                        combined_loss = (BCE_concept.detach().item() + BCE_main.detach().item())/2 
                   
                   
                   
                        combinations_dict[i]['results'][run_i] = combined_loss
                    print('Simulation {} of {} for comibination: weight_decay: {}, lr: {}, batch_size: {}'.format(run_i, sims, weight_decay_i, lr_i, batch_size_i))



                elif method == 'JTT':

                    
                    # get the optimizer settings
                    lr = optimizer_info['standard_ERM_settings'][dataset]['lr']
                    weight_decay = optimizer_info['standard_ERM_settings'][dataset]['weight_decay']
                    batch_size = optimizer_info['standard_ERM_settings'][dataset]['batch_size']

                    # set the parameters for identifier
                    lr_identifier = lr_i
                    weight_decay_identifier = weight_decay_i
                    batch_size_identifier = batch_size_i

                    # define the loaders
                    d = data_obj.X_train.shape[1]
                    identifier_loader =  data_obj.create_loaders(batch_size=batch_size_identifier, workers=0, with_concept=False,which_dependent='main', include_weights=balanced_training_main, train_weights=main_weights_train, val_weights=main_weights_val)



                    # get the model
                    identifier_model = return_linear_model(d, 
                                                identifier_loader,
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
                                                model_name=dataset+'_select_param_identifier_model')
                    
                                    # get predictions of the identifier model
                    y_m_pred_identifier_train = identifier_model(X_train)
                    y_m_pred_identifier_val = identifier_model(X_val)

                    # get the JTT weights
                    JTT_weights_train = get_JTT_weights(y_m_train, y_m_pred_identifier_train, weight_misclassified=weight_misclassified_i)
                    JTT_weights_val = get_JTT_weights(y_m_val, y_m_pred_identifier_val, weight_misclassified=weight_misclassified_i)
                    
                   
                    # reset the data objects
                    data_obj.reset_X(X_train, X_val, batch_size=batch_size, reset_X_objects=True, include_weights=balanced_training_main, train_weights = main_weights_train, val_weights = main_weights_val, add_indeces=True)
                    d = X_train.shape[1]

                    # Train the model to get JTT
                    JTT_model = return_linear_model(d,
                                                    data_obj.main_loader,
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
                                                    model_name=dataset+'_JTT_model', 
                                                    weights_BCE_train=JTT_weights_train, 
                                                    weights_BCE_val = JTT_weights_val)
                                                    


                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    y_m_val = data_obj.y_m_val
                    y_m_val_pred = JTT_model(X_val)

                    if metric == 'WG':
                        accuracy_per_group, _ =  get_acc_per_group(y_m_val_pred, y_m_val, y_c_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy

                    else:

                        if eval_balanced:
                            BCE_main = torch_calc_BCE_weighted(y_m_val, y_c_val,y_m_val_pred, device, type_dependent='main', reduce='mean')
                        else:
                            BCE_main = torch_calc_BCE(y_m_val,y_m_val_pred, device)
                        combinations_dict[i]['results'][run_i] = BCE_main.detach().item()

                elif method == 'GDRO':

                    # get the data
                    d = X_train.shape[1]
                    start_model_loader = data_obj.create_loaders( batch_size_i, 0, shuffle=True, with_concept=False, which_dependent='main', include_weights=False, train_weights = None, val_weights = None)

                    # Train the model to get main
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
                                                                model_name=dataset+'_start_model',
                                                                GDRO=False,
                                                                )

                    # get the state dict
                    start_model_state_dict = start_model.state_dict()
                        

                    
                    # reset the data objects
                    main_weights_train, main_weights_val = data_obj.get_group_weights()
                    GDRO_loaders = data_obj.create_loaders( batch_size_i, 0, shuffle=True, with_concept=True, which_dependent='main', include_weights=True, train_weights = main_weights_train, val_weights = main_weights_val, concept_first=False, add_indeces=False)
                        


                    # Train the model to get main
                    set_seed(seed)
                    d = data_obj.X_train.shape[1]
                    GDRO_model = return_linear_model(d, 
                                                            GDRO_loaders,
                                                            device,
                                                            solver = solver,
                                                            lr=lr_i,
                                                            per_step=per_step,
                                                            tol = optimizer_settings['tol'],
                                                            early_stopping = early_stopping,
                                                            patience = optimizer_settings['patience'],
                                                            epochs = epochs,
                                                            bias=True,
                                                            weight_decay=weight_decay_i, 
                                                            model_name=dataset+'_GDRO_main',
                                                            GDRO=True,
                                                            C=C_i,
                                                            eta_g=eta_g_i,
                                                            init_state_dict=start_model_state_dict,)
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    y_m_val = data_obj.y_m_val
                    y_c_val = data_obj.y_c_val
                    y_m_val_pred = GDRO_model(X_val)

                    if metric == 'WG':
                        accuracy_per_group, _ =  get_acc_per_group(y_m_val_pred, y_m_val, y_c_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy

                    else:

                        if eval_balanced:
                            BCE_main = torch_calc_BCE_weighted(y_m_val, y_c_val,y_m_val_pred, device, type_dependent='main', reduce='mean')
                        else:
                            BCE_main = torch_calc_BCE(y_m_val,y_m_val_pred, device)
                        combinations_dict[i]['results'][run_i] = BCE_main.detach().item()


                    print('BCE main: ', BCE_main.detach().item())
                    print('y_m_val (first 5): ', y_m_val[:5])
                    print('y_m_val_pred (first 5): ', y_m_val_pred[:5])
                    print('y_c_val (first 5): ', y_c_val[:5])


                elif method == 'INLP':
                    
                    # define the loaders
                    d = data_obj.X_train.shape[1]
                    concept_loader =  data_obj.create_loaders(batch_size=batch_size_i, workers=0, with_concept=False,which_dependent='concept', include_weights=balanced_training_concept, train_weights=concept_weights_train, val_weights=concept_weights_val)


                    # get the model
                    concept_model = return_linear_model(d, 
                                                concept_loader,
                                                device,
                                                solver = solver,
                                                lr=lr_i,
                                                per_step=per_step,
                                                tol = optimizer_settings['tol'],
                                                early_stopping = early_stopping,
                                                patience = optimizer_settings['patience'],
                                                epochs = epochs,
                                                bias=True,
                                                weight_decay=weight_decay_i, 
                                                model_name=dataset+'_select_param_concept_model')
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    y_c_val = data_obj.y_c_val
                    y_c_val_pred = concept_model(X_val)


                    if metric == 'WG':

                        accuracy_per_group, _ =  get_acc_per_group(y_c_val_pred, y_c_val, y_m_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy
                       
                    else:

                        if eval_balanced:
                            BCE_concept = torch_calc_BCE_weighted(y_m_val, y_c_val,y_c_val_pred, device, type_dependent='concept', reduce='mean')
                        else:
                            BCE_concept = torch_calc_BCE(y_c_val,y_c_val_pred, device)
                        combinations_dict[i]['results'][run_i] = BCE_concept.detach().item()

                
                elif method == 'ERM' or method == 'SUBG':
                    
                    # define the loaders
                    d = data_obj.X_train.shape[1]
                    if balanced_training_main or group_weighted:
                        include_weights = True
                    else:
                        include_weights = False
                    main_loader =  data_obj.create_loaders(batch_size=batch_size_i, workers=0, with_concept=False,which_dependent='main', include_weights=include_weights, train_weights=main_weights_train, val_weights=main_weights_val)


                    # get the model
                    main_model = return_linear_model(d, 
                                                main_loader,
                                                device,
                                                solver = solver,
                                                lr=lr_i,
                                                per_step=per_step,
                                                tol = optimizer_settings['tol'],
                                                early_stopping = early_stopping,
                                                patience = optimizer_settings['patience'],
                                                epochs = epochs,
                                                bias=True,
                                                weight_decay=weight_decay_i, 
                                                model_name=dataset+'_select_param_main_model')
                    
                    # calculate the binary cross-entropy of the concept model
                    X_val = data_obj.X_val
                    y_m_val = data_obj.y_m_val
                    y_m_val_pred = main_model(X_val)

                    if metric == 'WG':
                        accuracy_per_group, _ =  get_acc_per_group(y_m_val_pred, y_m_val, y_c_val)
                        worst_group_accuracy = np.min(accuracy_per_group)
                        combinations_dict[i]['results'][run_i] = -worst_group_accuracy

                    else:

                        if eval_balanced:
                            BCE_main = torch_calc_BCE_weighted(y_m_val, y_c_val,y_m_val_pred, device, type_dependent='main', reduce='mean')
                        else:
                            BCE_main = torch_calc_BCE(y_m_val,y_m_val_pred, device)
                        combinations_dict[i]['results'][run_i] = BCE_main.detach().item()


        i += 1
      

    

        

    # print the results at end
    print('-----------------------------------')

    overall_results = {}
    best_combination = None
    best_score = math.inf
    one_se_rule_combination = None
    best_score_plus_se = math.inf
    score_of_one_se_rule = math.inf
    weight_decay_best = None
    learning_rate_best = None
    batch_size_best = None



    for i in range(n_combinations):
        result_combination = combinations_dict[i]
        weight_decay_i = result_combination['param']['weight_decay']
        lr_i = result_combination['param']['lr']

        print('Average of BCE concept/main: ', np.mean(result_combination['results']))
        print('Standard deviation of BCE concept/main: ', np.std(result_combination['results']))
        print('For combination: {}'.format(result_combination['param']))

        avg_score = np.mean(result_combination['results'])
        std_score = np.std(result_combination['results'])
        se_score = std_score/np.sqrt(sims)

        if avg_score < best_score:
            best_combination = result_combination['param']
            best_score = avg_score
            best_score_plus_se = avg_score + se_score
            weight_decay_best = weight_decay_i
            learning_rate_best = lr_i
            print('current best score combination: {}'.format(best_combination))

        
        if (batch_size_i == batch_size_best) and (lr_i == learning_rate_best) and (weight_decay_i > weight_decay_best):

            if avg_score < best_score_plus_se and avg_score > best_score:
                print('-----------------------------------')
                print('combination based on one standard error rule')
                print(result_combination['param'])
                print('-----------------------------------')
                one_se_rule_combination = result_combination['param']
                score_of_one_se_rule = avg_score 

        

        entry = {'parameters': result_combination['param'], 'avg': avg_score, 'std': std_score, 'se': se_score}
        overall_results[i] = entry

    print('-----------------------------------')
    print('Overall results: {}'.format(overall_results))
    

    if one_se_rule_combination is None:
        print('No one standard error rule combination found - no other combination within one SE and higher weight_decay ')
    print('-----------------------------------')
    print('Best combination: {}'.format(best_combination))
    print('Best score: {}'.format(best_score))
    print('-----------------------------------')
    print('One standard error rule combination: {}'.format(one_se_rule_combination))
    print('Best score plus standard error: {}'.format(score_of_one_se_rule))



          
        






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='JSE', help='Method to use')
    parser.add_argument("--param_name", default="lr", help="dataset to use")
    parser.add_argument("--spurious_ratio", type=float, default=0.5, help="ratio of spurious features")
    parser.add_argument("--sims", type=int, default=10, help="number of simulations to run")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")
    parser.add_argument("--dataset_setting", type=str, default="default", help="dataset setting to use")
    parser.add_argument("--demean", type=str,  help="whether to demean the data")
    parser.add_argument("--pca", type=str, default=False, help="whether to use pca")
    parser.add_argument("--k_components", type=int, default=10, help="number of components to use for pca")
    parser.add_argument("--batch_size", help="batch size to use")
    parser.add_argument("--solver", type=str, default="SGD", help="solver to use")
    parser.add_argument("--lr",  help="learning rate to use")
    parser.add_argument("--weight_decay",  help="weight decay to use")
    parser.add_argument("--weight_misclassified",  type=str, default=1, help="weight misclassified to use")
    parser.add_argument("--early_stopping", type=str, help="whether to use early stopping")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to use")
    parser.add_argument("--per_step", type=int, default=1, help="per steps, print the loss")
    parser.add_argument("--seed", type=int, default=0, help="seed to use")
    parser.add_argument("--device_type", type=str, default="cuda", help="device to use")
    parser.add_argument("--group_weighted", type=str, default='False', help="whether to use group-weighted for main task")
    parser.add_argument("--use_smaller_set", type=str, default='False', help="whether to use validation set")
    parser.add_argument("--n_for_estimation", type=str, default='all', help="whether to use validation set")
    parser.add_argument("--C", type=str, default='0', help="C for GDRO")
    parser.add_argument("--eta_g", type=str, default='0.1', help="eta_g for GDRO")
    parser.add_argument('--single_finetuned_model', type=str, default='False', help='whether to use a single finetuned model')
    
    args = parser.parse_args()
    dict_arguments = vars(args)

    method = dict_arguments['method']
    param_name = Convert(dict_arguments["param_name"], str)


    sims = dict_arguments["sims"]
    dataset = dict_arguments["dataset"]
    spurious_ratio = dict_arguments["spurious_ratio"]
    dataset_setting = dict_arguments["dataset_setting"]
    demean = str_to_bool(dict_arguments["demean"])
    pca = str_to_bool(dict_arguments["pca"])
    k_components = dict_arguments["k_components"]

    
    solver = dict_arguments["solver"]

    if 'weight_decay' in param_name:
        weight_decay = Convert(dict_arguments["weight_decay"], float)
    else:
        weight_decay = float(dict_arguments["weight_decay"])

    if 'batch_size' in param_name:
        batch_size = Convert(dict_arguments["batch_size"], int)
    else:
        batch_size = int(dict_arguments["batch_size"])

    if 'lr' in param_name:
        lr = Convert(dict_arguments["lr"], float)
    else:
        lr = float(dict_arguments["lr"])

    if 'weight_misclassified' in param_name:
        weight_misclassified = Convert(dict_arguments["weight_misclassified"], float)
    else:
        weight_misclassified = float(dict_arguments["weight_misclassified"])

    if 'C' in param_name:
        C = Convert(dict_arguments["C"], float)
    else:
        C = float(dict_arguments["C"])
    
    if 'eta_g' in param_name:
        eta_g = Convert(dict_arguments["eta_g"], float)
    else:
        eta_g = float(dict_arguments["eta_g"])


    
    early_stopping = str_to_bool(dict_arguments["early_stopping"])
    epochs = dict_arguments["epochs"]
    per_step = dict_arguments["per_step"]
    device_type = dict_arguments["device_type"]
    seed = dict_arguments["seed"]
    group_weighted = str_to_bool(dict_arguments["group_weighted"])
    use_smaller_set = str_to_bool(dict_arguments["use_smaller_set"])
    n_for_estimation = dict_arguments["n_for_estimation"]
    single_finetuned_model = str_to_bool(dict_arguments["single_finetuned_model"])

    main(method, param_name, sims, dataset, spurious_ratio, dataset_setting, demean, pca, k_components,   batch_size, solver, lr, weight_decay, early_stopping, epochs,  per_step, device_type, seed, group_weighted=group_weighted, weight_misclassified=weight_misclassified, use_smaller_set=use_smaller_set, n_for_estimation=n_for_estimation, C=C, eta_g=eta_g, single_finetuned_model=single_finetuned_model)
