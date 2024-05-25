#!/bin/bash

# Define the base directory
baseDir="/Data/mellah.adib/cheese_classification_challenge"

# Define the list of cheese names
cheeseNames=(
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
    command="accelerate launch $baseDir/diffusers/examples/dreambooth/train_dreambooth.py --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 --output_dir=\"$outputDir\" \
            --checkpointing_steps=50 --mixed_precision=fp16 --instance_data_dir=\"$instanceDir\" --instance_prompt=\"a photo of a $cheeseName cheese\" --use_8bit_adam \
            --gradient_checkpointing --snr_gamma=5.0 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=2e-6 --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=400"

    # Execute the command
    echo "Training model for $cheeseName..."
    eval $command
done
