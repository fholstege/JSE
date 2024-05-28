
import numpy as np
import pandas as pd
import torch


from JSE.data import *
from JSE.settings import data_info, optimizer_info
from JSE.models import *
from JSE.training import *
from JSE.RLACE import *
from JSE.helpers import *

import argparse
import os
import sys

from JSE.settings import data_info

def main(method, file_with_images, weights_filename, seed, device_type='cpu', spurious_ratio=0.0):


    
    # create the images
    image_resolution = data_info['celebA']['black_and_white']['image_resolution']
    main_task_name = data_info['celebA']['black_and_white']['main_task_name']
    concept_name = data_info['celebA']['black_and_white']['concept_name']
    train_size = data_info['celebA']['black_and_white']['train_size']
    val_size = data_info['celebA']['black_and_white']['val_size']
    test_size = data_info['celebA']['black_and_white']['test_size']
    sample_n_images = train_size + val_size + test_size


    target_resolution = (image_resolution, image_resolution)
    transform_func = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize(target_resolution), 
                                        transforms.ToTensor(),
                                        torch.flatten])
    
    
    # read the celebA metadata
    df_metadata = pd.read_csv('datasets/celebA/raw_input/list_attr_celeba.txt', sep='\s+', skiprows=1)
    df_metadata = df_metadata.reset_index(drop=False)
    df_metadata.columns = ['img_filename'] + list(df_metadata.columns[1:])
    df_metadata['y'] = df_metadata[main_task_name].replace({-1:0})

    # read the file with images
    df_images = pd.read_csv(file_with_images)
    list_images_to_show = df_images['img_filename'].tolist()

    # get the images
    df_metadata_to_show = df_metadata[df_metadata['img_filename'].isin(list_images_to_show)]

    # get the images
    folder_images = 'datasets/celebA/raw_input/'
    X, _ = load_images(df_metadata_to_show, transform_func, folder_images+'images', '')
    X = X.squeeze(-1)

    # get the min and max values of X
    min_X = torch.min(X)
    max_X = torch.max(X)
    print('min_X: ', min_X)
    print('max_X: ', max_X)


   


   
    # load the weights
    with open(weights_filename, 'rb') as fp:
            result_dict = pickle.load(fp)

    
    print('Parameters of the model: ', result_dict['parameters'])


    # get the weights
    if method == 'INLP' or method == 'JSE':
        
        # get the concept subspace
        V_c = result_dict['V_c']

        # get the projection matrix
        P_c_orth = torch.eye(V_c.shape[0]) - create_P(V_c)

        # transform the data
        X_after_proj = torch.matmul(X, P_c_orth)

    elif method == 'RLACE':
        
        # get the projection matrix
        P_c_orth = result_dict['P_c_orth']

        # get the V_k
        V_k = result_dict['V_k_train']

        # transform the data
        X_k = torch.matmul(X, V_k)

        # transform the data
        X_after_proj_k = torch.matmul(X_k, P_c_orth)

        # transform back to original space
        X_after_proj = torch.matmul(X_after_proj_k, V_k.T)


    elif method == 'LEACE':
         
        # get the eraser
        eraser = result_dict['Eraser']

        # transform the data
        X_after_proj = eraser(X)

    

    min_X_after_proj = torch.min(X_after_proj)
    max_X_after_proj = torch.max(X_after_proj)
    print('min_X_after_proj: ', min_X_after_proj)
    print('max_X_after_proj: ', max_X_after_proj)

         

    folder_save = 'datasets/celebA/illustrated_images/{}/'.format(method)

    all_X_illustrate_diff = []
    X_diff = abs(X - X_after_proj)
  
    for i in range(len(list_images_to_show)):

            # get the index of the images
            filename = list_images_to_show[i]
            extra_info_img = '{}_seed_{}_method_{}_spurious_ratio_{}'.format(filename[:-4], seed,  method, int(100*spurious_ratio))
            print('Checking image: ', filename)
            bool_file = df_metadata_to_show['img_filename'] == filename
            i_to_show = np.where(bool_file.tolist())[0][0]

            # select this original image, and show it
            X_illustrate = X[i_to_show:(i_to_show+1), :]
            get_image_from_vec(X_illustrate, image_resolution, save=True, img_name=extra_info_img + '_original_image_file', show=False, folder=folder_save, cmap = 'gray', vmin=min_X_after_proj, vmax=max_X_after_proj)
            min_original_image = torch.min(X_illustrate)
            max_original_image = torch.max(X_illustrate)

            # select the image after projection, and show it
            X_illustrate_after_proj = X_after_proj[i_to_show:(i_to_show+1), :]
            X_illustrate_after_proj_norm = (X_illustrate - min_original_image)/(max_original_image - min_original_image)

            get_image_from_vec(X_illustrate_after_proj, image_resolution, save=True, img_name=extra_info_img+ '_after_image_file', show=False, folder=folder_save, cmap = 'gray', vmin=min_X_after_proj, vmax=max_X_after_proj)
                
            # show the difference in pixel values between the two
            X_illustrate_diff = X_diff[i_to_show:(i_to_show+1), :]
            all_X_illustrate_diff.append(X_illustrate_diff)
            get_image_from_vec(X_illustrate_diff, image_resolution, cmap='Reds', save=True, img_name=extra_info_img+'_diff_image_file', show=False, folder=folder_save)


    all_X_illustrate_diff = np.array(all_X_illustrate_diff)
    # get var for each pixel
    var_X_illustrate_diff = np.var(all_X_illustrate_diff, axis=0)
    print('var_X_illustrate_diff: ', var_X_illustrate_diff)
    # print the max values of the variances
    print('max variances: ', np.max(var_X_illustrate_diff, axis=1))


# method, file_with_images, weights_filename
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    parser.add_argument('--file_with_images')
    parser.add_argument('--weights_filename')
    parser.add_argument('--spurious_ratio', default=0.0, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device_type', default='cpu')


   
    args = parser.parse_args()

    method = args.method
    file_with_images = args.file_with_images
    weights_filename = args.weights_filename
    spurious_ratio = args.spurious_ratio
    seed = args.seed
    device_type = args.device_type


   
   

    main(method, file_with_images, weights_filename, seed, device_type=device_type, spurious_ratio=spurious_ratio)