import torch
from transformers import StableDiffusionForImageGeneration, AutoFeatureExtractor

# Load the pre-trained Stable Diffusion model
model = StableDiffusionForImageGeneration.from_pretrained("stabilityai/stable-diffusion-1-4")

# Load the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("stabilityai/stable-diffusion-1-4")

# Define the input prompt
prompt = "A beautiful sunset on the beach"

# Convert the prompt to a tensor
input_ids = torch.tensor([[model.decoder.encode(prompt)]).long())

# Generate the image
image = model.generate(input_ids=input_ids, num_beams=4, max_length=256)

# Convert the generated image to a PIL image
pil_image = feature_extractor.decode(image[0])

# Save the generated image to a file
pil_image.save("generated_image.png")
