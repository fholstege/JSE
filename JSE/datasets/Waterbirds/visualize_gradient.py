
import numpy as np
import pandas as pd
import torch

from JSE.data import *
from JSE.settings import data_info, optimizer_info
from JSE.models import *
from JSE.training import *
from JSE.RLACE import *

from captum.attr import InputXGradient, IntegratedGradients, NoiseTunnel, visualization, Saliency, GuidedGradCam, LayerGradCam, LayerActivation, LayerAttribution
from captum.attr import visualization as viz

import argparse
import os
import sys

import torch
import torchvision 

from create_waterbird_embeddings import transform_cub

def main(file_with_images, weights_filename, method):

    # read the file with images
    df = pd.read_csv(file_with_images)

    # load the resnet50 model
    model = torchvision.models.resnet50(pretrained = True)
    model = model.eval()

    # get embedding_creator, which just removes the last linear layer of resnet50
    model_embeddings =  embedding_creator(model)
    model_embeddings = model_embeddings.eval()

        

    class combined_model(torch.nn.Module):
        def __init__(self, model_embeddings, W, V_c=None, V_k=None, demean=True, input_mean=None):
            super(combined_model, self).__init__()
            self.model_embeddings = model_embeddings
            self.W = W
            self.demean = demean

            if self.demean:
                self.input_mean = input_mean
                
            if V_c is not None:
                self.d = V_c.shape[0]
                self.P_c_orth = torch.eye(self.d) - create_P(V_c)
            else:
                self.P_c_orth = None
                
            if V_k is not None:
                self.V_k = V_k

        def forward(self, x):
            classifier_input = self.model_embeddings(x)

            if self.demean:
                classifier_input = classifier_input - self.input_mean

            if self.V_k is not None:
                classifier_input = torch.matmul(classifier_input, self.V_k)

            if self.P_c_orth is not None:
                classifier_input = torch.matmul(classifier_input, self.P_c_orth)
                
            

            y = self.W(classifier_input)
            return y
            


    # load the weights
    with open(weights_filename, 'rb') as fp:
            result_dict = pickle.load(fp)

            # load the weights, bias
            param_W = result_dict['param_W']
            param_W['weight'] = param_W.pop('linear_layer.weight')
            param_W['bias'] = param_W.pop('linear_layer.bias')

            print('Added parameters')
            if 'V_c' in result_dict.keys():
                V_c = result_dict['V_c']
            else:
                V_c = None

            if method == 'RLACE':
                P_c_orth = result_dict['P_c_orth']

            if result_dict['V_k_train'] is not None:
                V_k = result_dict['V_k_train']
            else:
                V_k = None

            if 'X_train_mean'in result_dict.keys():
                if result_dict['X_train_mean'] is not None:
                    demean = True
                    input_mean = result_dict['X_train_mean']
                else:
                    demean = False
                    input_mean = None
            else:
                demean = False
                input_mean = None

            

            
        
    # set the weights, bias
    # define a linear layer with 2048 inputs, 1 output
    if V_k is not None:
        d = V_k.shape[1]
    else:
        d = 2048
        
    W = torch.nn.Linear(d, 1, bias=True)
    W.load_state_dict(param_W)
    

    # create combined model
    combined_model_obj = combined_model(model_embeddings, W, V_c=V_c, V_k=V_k, demean=demean, input_mean=input_mean)
    combined_model_obj = combined_model_obj.eval()

    if method == 'RLACE':
        combined_model_obj.P_c_orth = P_c_orth

    total_correct = 0


    # create pd.dataframe with the images, and the method
    df_predicted = pd.DataFrame(columns=['img_filename', 'y', 'place', 'method', 'predicted_y'])
    df_predicted['img_filename'] = df['img_filename']
    df_predicted['y'] = df['y']
    df_predicted['place'] = df['place']
    df_predicted['method'] = method


    with torch.no_grad():
        # loop through the images
        for index, row in df.iterrows():
            image_filename = row['img_filename']
            image_filename = 'images/' + image_filename


            # convert to tensor
            image =  Image.open(image_filename).convert('RGB')


            # create data_obj
            data_obj = Waterbird_Dataset()
            
            # change to (224, 224) resolution
            resolution_waterbird = (224, 224)

            # get transform function
            transform_func = data_obj.transform_cub(resolution_waterbird)
            image_for_model = transform_func(image)

            # apply transform function for the original image to visualize
            resolution_waterbird = (224, 224)
            scale = 256.0/224.0
            resize_func = transforms.Compose([
                                                    transforms.Resize((int(resolution_waterbird[0]*scale), int(resolution_waterbird[1]*scale))), # resize
                                                    transforms.CenterCrop(resolution_waterbird),  # crop the center
                                                    transforms.ToTensor(), # turn to tensor
            ])
            image_for_viz = resize_func(image)

            # add batch dimension
            image_for_model = image_for_model.unsqueeze(0)

            # apply Variable() to the image tensor, set requires_grad=True
            image_for_viz = Variable(image_for_viz, requires_grad=False)
            image_for_model = Variable(image_for_model, requires_grad=False)

            print('The shape of the image is: ', image_for_model.shape)


            
            output = combined_model_obj(image_for_model)
            

            selected_layer = combined_model_obj.model_embeddings.feature_extractor[-2][-1]
            #LayerActivation_calc = LayerActivation(combined_model_obj, selected_layer)
            #activation = LayerActivation_calc.attribute(image_for_model, attribute_to_layer_input=False)

            GCam = LayerGradCam(combined_model_obj, selected_layer)
            attribution = GCam.attribute(image_for_model, 0, attribute_to_layer_input=False)
            upsampled_attribution = LayerAttribution.interpolate(attribution,image_for_model.shape[2:])

            upsampled_attribution_for_viz = upsampled_attribution.squeeze(0).permute(1,2,0).detach().numpy()
            original_image_resized_for_viz = image_for_viz.squeeze(0).permute(1,2,0).detach().numpy()
            
            fig, ax = viz.visualize_image_attr_multiple(upsampled_attribution_for_viz ,original_image=original_image_resized_for_viz,signs=["all",  'all' ],methods=["original_image",  'blended_heat_map'], outlier_perc=2, use_pyplot=True)

            # create filename that combines the image filename and the weights filename
            weight_filename_end = weights_filename.split('/')[-1]
            image_filename_end = image_filename.split('/')[-1]

            # remove from both the .jpg
            weight_filename_end = weight_filename_end.split('.')[0]
            image_filename_end = image_filename_end.split('.')[0]

            # combine the two
            filename = image_filename_end + '_' + weight_filename_end
            
            # plt.savefig('gradient_viz/' + filename + '.png', dpi=300)
            print('The image filename is: ', image_filename)
            print('Output for this image is: ', torch.sigmoid(output))
            print('The predicted class is: ', torch.round(torch.sigmoid(output)))
            print('The true class is: ', row['y'])
            print('The background is: ', row['place'])

            correct = torch.round(torch.sigmoid(output)) == row['y']
            total_correct += correct
            print('The accuracy so far is: ', total_correct/(index+1))

            df_predicted.loc[index, 'predicted_y'] = torch.round(torch.sigmoid(output)).item()


            # save the figure
            fig.savefig('gradient_viz/{}/'.format(method) + filename + '.png', dpi=300)


    # save the dataframe
    df_predicted.to_csv('gradient_viz/{}/'.format(method) + 'df_predicted.csv', index=False)


   

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_with_images')
    parser.add_argument('--weights_filename')
    parser.add_argument('--method')


   
    args = parser.parse_args()

    file_with_images = args.file_with_images
    weights_filename = args.weights_filename
    method = args.method
   

   
   

    main(file_with_images, weights_filename, method)