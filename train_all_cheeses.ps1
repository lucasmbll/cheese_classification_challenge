# Define the base directory
$baseDir = "C:/Users/adib4/OneDrive/Documents/Travail/X/MODAL DL/cheese_classification_challenge"

# Define the list of cheese names
$cheeseNames = @(
    "BRIE DE MELUN",
    "CAMEMBERT",
    "EPOISSES",
    "FOURME D’AMBERT",
    "RACLETTE",
    "MORBIER",
    "SAINT-NECTAIRE",
    "POULIGNY SAINT- PIERRE",
    "ROQUEFORT",
    "COMTÉ",
    "CHÈVRE",
    "PECORINO",
    "NEUFCHATEL",
    "CHEDDAR",
    "BÛCHETTE DE CHÈVRE",
    "PARMESAN",
    "SAINT- FÉLICIEN",
    "MONT D’OR",
    "STILTON",
    "SCARMOZA",
    "CABECOU",
    "BEAUFORT",
    "MUNSTER",
    "CHABICHOU",
    "TOMME DE VACHE",
    "REBLOCHON",
    "EMMENTAL",
    "FETA",
    "OSSAU- IRATY",
    "MIMOLETTE",
    "MAROILLES",
    "GRUYÈRE",
    "MOTHAIS",
    "VACHERIN",
    "MOZZARELLA",
    "TÊTE DE MOINES",
    "FROMAGE FRAIS"
)

# Iterate over each cheese in the list
foreach ($cheeseName in $cheeseNames) {
    # Set the environment variables
    $instanceDir = "$baseDir/dataset/val/$cheeseName"
    $outputDir = "$baseDir/db_models/$cheeseName"
    $classDataDir = "$baseDir/class_data/$cheeseName"

    # Create the output directory if it doesn't exist
    if (-Not (Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir
    }

    # Construct the command with quoted paths and additional parameters
    $command = "accelerate launch '$baseDir/diffusers/examples/dreambooth/train_dreambooth.py' --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5' --output_dir='$outputDir' --checkpointing_steps=50 --mixed_precision='fp16' --instance_data_dir='$instanceDir' --instance_prompt='a photo of a $cheeseName cheese' --use_8bit_adam --gradient_checkpointing --with_prior_preservation --prior_loss_weight=1.0 --class_data_dir='$classDataDir' --class_prompt='a photo of a $cheeseName cheese'"

    # Execute the command
    Write-Host "Training model for $cheeseName..."
    Invoke-Expression $command
}
