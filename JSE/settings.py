

optimizer_info = {
    'All': {
        'tol': 1e-3,
        'patience': 5
    },

    'standard_ERM_settings': {
        'Waterbirds': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.01, 
            'batch_size': 128 
        },
        'multiNLI': {
            'weight_decay': 1, 
            'lr': 0.01,
            'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.00,
            'lr': 0.1,
            'batch_size': 128

        }
    },

    'standard_LEACE_settings': {
        'Waterbirds': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.01, 
            'batch_size': 128 
        },
        'multiNLI': {
            'weight_decay': 1, 
            'lr': 0.01,
            'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.00,
            'lr': 0.1,
            'batch_size': 128

        }
    },

    'standard_ERM_settings_GW':  {
        'Waterbirds': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.01, 
            'batch_size': 128 
        },
        'multiNLI': {
            'weight_decay': 0.1, 
            'lr': 0.01,
            'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.00,
            'lr': 0.1,
            'batch_size': 128

        }
    },
    'standard_ERM_settings_SUBG':{
        'Waterbirds':{
            'weight_decay':0.01,
            'lr':0.01,
            'batch_size':64
        },
        'celebA':{
            'weight_decay':0.01,
            'lr': 0.01,
            'batch_size':64
        },
        'multiNLI':{
            'weight_decay':0.1,
            'lr':0.0001,
            'batch_size':64
        },
        'Toy':{
            'weight_decay':0.00,
            'lr':0.1,
            'batch_size':64
        }

    },


    'standard_GDRO_settings': {
        'Waterbirds': {
            'weight_decay': 0.1,
            'lr': 0.001,
            'batch_size': 128,
            'C':1
        },
        'celebA': {
            'weight_decay': 0.001,
            'lr': 0.001,
            'batch_size': 128,
            'C':2
        },
        'multiNLI': {
            'weight_decay': 1,
            'lr': 0.0001,
            'batch_size': 128,
            'C':5
        },
        'Toy': {
            'weight_decay': 0,
            'lr': 0.001,
            'batch_size': 128,
            'C': 5
        }
    },
    'standard_GDRO_settings_smaller_dataset': {
        'Waterbirds': {
            'weight_decay': 0.1,
            'lr': 0.001,
            'batch_size': 64,
            'C':1
        },
        'celebA': {
            'weight_decay': 0.001,
            'lr': 0.001,
            'batch_size': 64,
            'C':2
        },
        'multiNLI': {
            'weight_decay': 1,
            'lr': 0.0001,
            'batch_size': 64,
            'C':5
        },
        'Toy': {
            'weight_decay': 0,
            'lr': 0.001,
            'batch_size': 64,
            'C': 5
        }
    },


    'standard_ERM_settings_smaller_dataset': {
        'Waterbirds': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 64
        },
        'celebA': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 64
        },
        'multiNLI': {
            'weight_decay': 1,
            'lr': 0.01,
            'batch_size': 64
        },
        'Toy': {
            'weight_decay': 0.00,
            'lr': 0.1,
            'batch_size': 64
        }
    },

    'standard_JTT_settings': {
        'Waterbirds': {
            'weight_decay': 1,
            'lr': 0.001,
            'batch_size': 128,
            'weight_misclassified':2

        },
        'celebA': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128,
            'weight_misclassified':5
        },
        'multiNLI': {
            'weight_decay': 0.001,
            'lr': 0.00001,
            'batch_size': 128,
            'weight_misclassified':10

        },
        'Toy': {
            'weight_decay': 0.0,
            'lr': 0.0001,
            'batch_size': 128,
            'weight_misclassified':2
        }
    },

    'standard_JSE_settings': {
        'Waterbirds': {
            'weight_decay': 0.001,
            'lr': 0.001, 
            'batch_size': 128
        },
        'celebA': {
            'weight_decay': 0.001, 
            'lr': 0.001, 
            'batch_size': 128  
        },
        'multiNLI': {
             'weight_decay': 0.01,
               'lr': 0.01, 
               'batch_size': 128
        },
        'Toy': {
            'weight_decay': 0.0,
            'lr': 0.01,
            'batch_size': 128
        }
    },

    'standard_JSE_settings_smaller_dataset': {
        'Waterbirds': {
            'weight_decay': 0.001,
            'lr': 0.001,
            'batch_size': 64
        },
        'celebA': {
            'weight_decay': 0.001,
            'lr': 0.001,
            'batch_size': 64
        },
        'multiNLI': {
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 64
        },
        'Toy': {
            'weight_decay': 0.0,
            'lr': 0.01,
            'batch_size': 64
        }
    },

    'standard_INLP_settings':{
       'Toy': {
            'weight_decay': 0.0,
            'lr': 0.1,
            'batch_size': 128
        },
        'Waterbirds': {
            'weight_decay': 0.001,
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01,
            'lr': 0.01,
            'batch_size': 128
        },
        'multiNLI': {
            'weight_decay': 0.001, 
            'lr': 0.1,
            'batch_size': 128
        }

    
        
    },

    'standard_RLACE_settings': {
        'Toy': {
            'weight_decay': 0.0, 
            'lr': 0.1,
            'batch_size': 128
        },
        'Waterbirds': {
            'weight_decay': 0.001, 
            'lr': 0.01,
            'batch_size': 128
        },
        'celebA': { 
            'weight_decay': 0.01, 
            'lr': 0.01,
            'batch_size': 128
        },
        'multiNLI': {
            'weight_decay': 0.001,
            'lr': 0.1,
            'batch_size': 512

        }
    },
    
}



data_info = {
    'All': {
        'train_split': 0.8
    },
    'celebA': {
        'default': {
            'set_sample':False,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
             'adversarial':False,
            'combine_sets': False,
            'fine_tuned': False,
            'black_and_white_original': False,
            'second_concept_name':None,

        },
        'sampled_data': {
            'set_sample':True,
            'p_m':0.5,
            'train_size': 4500,
            'val_size': 2000,
            'test_size': 2000,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
             'adversarial':False,
             'finetuned': False,
            'combine_sets': True,
            'black_and_white_original': False,
            'second_concept_name':None,
        },
        'sampled_data_two_concepts': {
            'set_sample':True,
            'p_m':0.5,
            'train_size': 4500,
            'val_size': 2000,
            'test_size': 2000,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
            'second_concept_name':'Eyeglasses',
            'adversarial':False,
            'finetuned': True,
            'combine_sets': True,
            'black_and_white_original': False,
            'settings': 'kirichenko',
            'finetune_seed':2000,
            'early_stopping':False
            
        },

        'sampled_data_finetuned':{
            'set_sample':False,
            'train_size': 4500,
            'val_size': 2000,
            'test_size': 2000,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
            'adversarial':False,
            'combine_sets': True,
            'black_and_white_original': False,
            'early_stopping': False,
            'finetuned': True,
            'settings': 'kirichenko',
            'finetune_seed':2000,
            'second_concept_name':None

        },
        
        'sampled_data_adv': {
            'set_sample':False,
            'p_m':0.5,
            'train_size': 4500,
            'val_size': 2000,
            'test_size': 2000,
            'main_task_name': 'Blond_Hair',
            'concept_name': 'Female',
            'adversarial':True,
            'finetuned': True,
            'settings': 'kirichenko',
            'early_stopping': True,
            'black_and_white_original': False,
            'finetune_seed':1000,
            'second_concept_name':None,


            },
        'black_and_white': {
            'adversarial':False,
            'set_sample':False,
            'main_task_name': 'Eyeglasses',
            'concept_name':'Smiling',
            'train_size':8000,
            'val_size': 2000,
            'test_size': 2000,
            'black_and_white_original': True,
            'image_resolution': 50,
            'finetuned': False,
            'finetune_seed':None,
            'second_concept_name':None

        }
    },
    'Toy': {
    'default': {
        'n': 2000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0,
        'finetune_seed':None
      

        },

    'sample_size_1000':{
        'n': 1000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'sample_size_500':{
        'n': 500,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'sample_size_5000':{
        'n': 5000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'sample_size_10000':{
        'n': 10000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0.0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0
    },

    'non_orthogonal': {'n': 2000,
        'd': 20,
        'gamma_c': 3,
        'gamma_m': 3,
        'rho_c': 0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':15,
        'finetune_seed':None
      

        },
    'different_separability': {'n': 2000,
        'd': 20,
        'gamma_c': 6,
        'gamma_m': 2,
        'angle': 90,
        'rho_c': 0,
        'rho_m': 0,
        'intercept_concept': 0,
        'intercept_main': 0,
        'X_variance': 1,
        'angle':0

        },
    },

    'Waterbirds':{
        'default': {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
             'balance_main': False,
            'adversarial':False,
            'finetuned':False,
            'settings':None
        },
        'default_finetuned':
        {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
            'balance_main': False,
            'adversarial':False,
            'finetuned':True,
            'settings': 'kirichenko',
            'early_stopping': False,
            'finetune_seed': 2000
        },

        'default_adv': {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
            'balance_main': False,
            'adversarial':True,
            'finetuned':True,
            'settings': 'kirichenko',
            'early_stopping': True,
            'finetune_seed': 1000
             
        },
        'balanced': {
            'corr_test':'50',
            'image_type_train_val': 'combined',
            'image_type_test': 'combined',
            'model_name': 'resnet50',
            'balance_main': True,
            'adversarial':False,

        },

    },
    'multiNLI':{ 
        'default': {
            'finetune_param_type': 'adversarial_param',
            'embedding_type': 'CLS',
            'finetuned_BERT': 1,
            'train_size': 50000,
            'val_size': 5000,
            'binary_task': True,
            'dropout': 0.1,
            'early_stopping': True,
            'finetune_mode': 'CLS',
            'finetune_seed':1,
            'spurious_ratio_train': None
            
        },
        'from_50_50_sample': {
             'finetune_param_type': 'adversarial_param',
            'embedding_type': 'CLS',
            'finetuned_BERT': 1,
            'train_size': 50000,
            'val_size': 5000,
            'binary_task': True,
            'dropout': 0.1,
            'early_stopping': True,
            'finetune_mode': 'CLS',
            'finetune_seed':1,
            'spurious_ratio_train': 0.5,
            
        },
        'same_concept_value':{
            'finetune_param_type': 'adversarial_param',
            'embedding_type': 'CLS',
            'finetuned_BERT': 1,
            'train_size': 50000,
            'val_size': 5000,
            'binary_task': True,
            'dropout': 0.1,
            'early_stopping': True,
            'finetune_mode': 'CLS',
            'finetune_seed':2000,
            'spurious_ratio_train': 0.0
        }
    } 
}
