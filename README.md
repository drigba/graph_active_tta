How to run:
1. Set default values in the config directory under the predictor
2. Set parameters in the command line
   
    ```
    python main2.py  data.num_splits=5 model.num_inits=5 print_summary=True model.cached=False seed=42 predictor.augmentor.augmentations.0.type_=feature_noising predictor.augmentor.augmentations.0.p=0.4 predictor.augmentor.augmentations.1.type_=edge_dropout predictor.augmentor.augmentations.1.p=0.4 wandb.name=refactored_test_lc_fnoise_emask_0.4_0.4_hard_filter_scale_continous
   #This results in 2 augmentation functions, feature noising and edge drop, both with p=0.4, which will be called sequentially. This line of code does not set the predictor type or filtering.
    ```
    
4. Combination of options 1 and 2
