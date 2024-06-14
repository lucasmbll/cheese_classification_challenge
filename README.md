# Cheese Classification challenge
This codebase allows you to jumpstart the INF473V challenge.
The goal of this channel is to create a cheese classifier without any real training data.
You will need to create your own training data from tools such as Stable Diffusion, SD-XL, etc...

## Instalation

Cloning the repo:
```
git clone git@github.com:nicolas-dufour/cheese_classification_challenge.git
cd cheese_classification_challenge
```
Install dependencies:
```
conda create -n cheese_challenge python=3.10
conda activate cheese_challenge
pip install -r requirements.txt
```

Download the data from kaggle and copy them in the dataset folder
The data should be organized as follow: ```dataset/val```, ```dataset/test```. then the generated train sets will go to ```dataset/train/your_new_train_set```

## Using this codebase
This codebase is centered around 2 components: generating your training data and training your model.
Both rely on a config management library called hydra. It allow you to have modular code where you can easily swap methods, hparams, etc

### Training

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```

### Generating datasets
You can generate datasets with the following command

```
python generate.py
```

If you want to create a new dataset generator method, write a method that inherits from `data.dataset_generators.base.DatasetGenerator` and create a new config file in `configs/generate/dataset_generator`.
You can then run

```
python generate.py dataset_generator=your_new_generator
```

### VRAM issues
If you have vram issues either use smaller diffusion models (SD 1.5) or try CPU offloading (much slower). For example for sdxl lightning you can do

```
python generate.py image_generator.use_cpu_offload=true
```

## Create submition
To create a submition file, you can run 
```
python create_submition.py experiment_name="name_of_the_exp_you_want_to_score" model=config_of_the_exp
```

Make sure to specify the name of the checkpoint you want to score and to have the right model config


## Files of Adib and Lucas

* train_all_cheeses.sh : Training of DreamBooth for all cheeses (outputs 37 models) with parameter of the "command" variable. The data for training is the validation set. The output models are stored in db_models folder.
* ocr.py : to run OCR during submition creation
* submition_multiple_models.py : to make a submition with mutliple models
* zeroshot_sort.py was attempt to automate the sorting of generated images with clip in order to keep only best quality pictures.
* analysis_optuna.py : for hyperparameters optimization via Optuna
* augment_valdata.py : attempt for using data_augmentation on the val set to train DreamBooth
* cheese_contexts.txt : texte with context for prompt engineering
* cheese_ocr.txt : text that OCR should be able to recognize
* cheese_prompts.txt : more precised situation for prompt engineering
* db_to_retrain.txt : Dreambooths models that have to be trained again (because some cheese have not been well understood the first time).
* doubletune.py : for a second finetune phase
* generate_augmented.py and data_augmentation.py : data augmentation
