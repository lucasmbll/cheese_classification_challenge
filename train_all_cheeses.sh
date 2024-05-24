#!/bin/bash

# Define the base directory
baseDir="/Data/mellah.adib/cheese_classification_challenge"

# Define the list of cheese names
cheeseNames=(
    "MIMOLETTE"
    "MAROILLES"
    "GRUYÈRE"
    "MOTHAIS"
    "VACHERIN"
    "MOZZARELLA"
    "TÊTE DE MOINES"
    "FROMAGE FRAIS"
)

 

# Iterate over each cheese in the list
for cheeseName in "${cheeseNames[@]}"; do
    # Set the environment variables
    instanceDir="$baseDir/dataset/val/$cheeseName"
    outputDir="$baseDir/db_models/$cheeseName"
    classDataDir="$baseDir/class_data/$cheeseName"

    # Create the output directory if it doesn't exist
    if [ ! -d "$outputDir" ]; then
        mkdir -p "$outputDir"
    fi

    # Construct the command
    # command="accelerate launch '$baseDir/diffusers/examples/dreambooth/train_dreambooth_lora.py' --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' --output_dir='$outputDir' --checkpointing_steps=50 --mixed_precision='fp16' --instance_data_dir='$instanceDir' --instance_prompt='a photo of a $cheeseName cheese' --use_8bit_adam --gradient_checkpointing --with_prior_preservation --prior_loss_weight=1.0 --class_data_dir='$classDataDir' --class_prompt='a photo of a $cheeseName cheese'"
    command="accelerate launch '$baseDir/diffusers/examples/dreambooth/train_dreambooth_lora.py' --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' --output_dir='$outputDir' --checkpointing_steps=50 --mixed_precision='fp16' --instance_data_dir='$instanceDir' --instance_prompt='a photo of a $cheeseName cheese' --use_8bit_adam --gradient_checkpointing

    # Execute the command
    echo "Training model for $cheeseName..."
    eval "$command"
done
