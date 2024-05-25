#!/bin/bash

# Define the base directory
baseDir="/users/eleves-b/2022/lucas.mebille/Documents/cheese_challenge"

# Define the list of cheese names
cheeseNames=(
    RACLETTE
)

base_json_path="$baseDir/IP-Adapter/json_IPA"

# Iterate over each cheese in the list
for cheeseName in "${cheeseNames[@]}"; do
    # Set the environment variables
    instanceDir="$baseDir/dataset/train_db/$cheeseName"
    outputDir="$baseDir/IP-Adapter/cheese_models/$cheeseName"
    data_json_file="$base_json_path/$cheeseName.json"
    image_path="$baseDir/dataset/train_db/$cheeseName"

    # Create the output directory if it doesn't exist
    if [ ! -d "$outputDir" ]; then
        mkdir -p "$outputDir"
    fi
    
    # Construct the command
    command="accelerate launch --num_processes 8 --mixed_precision 'fp16' \
    $baseDir/IP-Adapter/tutorial_train.py \
    --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' \
    --image_encoder_path='$baseDir/IP-Adapter/models/image_encoder' \
    --data_json_file='$data_json_file' \
    --data_root_path='$image_path' \
    --mixed_precision='fp16' \
    --resolution=264 \
    --train_batch_size=8 \
    --dataloader_num_workers=4 \
    --learning_rate=1e-04 \
    --weight_decay=0.01 \
    --output_dir='$outputDir' \
    --save_steps=100"

    # Execute the command
    echo "Training model for $cheeseName..."
    eval "$command"
done
