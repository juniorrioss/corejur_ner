## Named Entity Recognition 

### In this repository, pre-processing, post-processing and model training are done.

### For using all features of this repo mainly modify config/settings.yaml and scripts/*.sh


### 0 - Put the conll files into folder data/raw

## PREPROCESSING
### Modify filename in config/setting.yaml and preprocessing args

```bash
  $ bash scripts/0-preprocess_dataset.sh
```

## POSTPROCESSING (FEATURE ENG.)
### The main attribute is `ratio_of_undersample_negative_sentences` for downsampling negative samples - Default to 0.5

```bash
  $ bash scripts/1-postprocess_dataset.sh
```


## TRAINING MODEL
### Modify hiperparameters in scripts/run_ner.sh
#### The importants ones are learning rate and batch size
```
  $ bash scripts/run_ner.sh
  
```

#### For using wandb it's recommeded to add a `.env` file
```.env
  WANDB_API_KEY = d7a2f3XXXXXXXXXXXXXXXXXX33e4 # API key from wandb settings
  WANDB_PROJECT = NER-COREJUR-V24 # Project name
```