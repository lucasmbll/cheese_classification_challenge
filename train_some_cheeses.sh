#!/bin/bash

# Define the base directory
baseDir="/Data/mellah.adib/cheese_classification_challenge"

# Define the list of cheeses to retrain with corresponding number of steps
declare -A cheesesToRetrain=(
    ["EPOISSES"]=800
    ["COMTÉ"]=800
    ["PECORINO"]=800
    ["SAINT- FÉLICIEN"]=800
    ["MONT D’OR"]=800
    ["SCARMOZA"]=1000
    ["CABECOU"]=800
    ["MUNSTER"]=1000
    ["REBLOCHON"]=800
    ["MAROILLES"]=1000
    ["GRUYÈRE"]=800
    ["MOTHAIS"]=800
    ["MOZZARELLA"]=800
    ["STILTON"]=500
    ["TÊTE DE MOINES"]=1000
)

# Function to train a stable diffusion model for a cheese with specific parameters
function TrainStableDiffusionModel {
    local cheeseName="$1"
    local numSteps="${cheesesToRetrain[$cheeseName]}"
    
    # Set the environment variables
    local instanceDir="$baseDir/dataset/val_sorted/$cheeseName"
    local outputDir="$baseDir/db_models/val_sorted/$cheeseName"

    # Create the output directory if it doesn't exist
    if [ ! -d "$outputDir" ]; then
        mkdir -p "$outputDir"
    fi

    # Construct the command with quoted paths and additional parameters
    local command="accelerate launch \"$baseDir/diffusers/examples/dreambooth/train_dreambooth.py\" --pretrained_model_name_or_path=\"runwayml/stable-diffusion-v1-5\" --output_dir=\"$outputDir\" \
            --checkpointing_steps=800 --mixed_precision=fp16 --instance_data_dir=\"$instanceDir\" --instance_prompt=\"a photo of a $cheeseName cheese\" --use_8bit_adam \
            --gradient_checkpointing --snr_gamma=5.0 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=2e-6 --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=$numSteps"

    # Execute the command
    echo "Training model for $cheeseName with $numSteps steps..."
    eval "$command"
}

# Train stable diffusion models for each cheese with specific parameters
for cheeseName in "${!cheesesToRetrain[@]}"; do
    # Call the function to train the stable diffusion model for the cheese
    TrainStableDiffusionModel "$cheeseName"
done
