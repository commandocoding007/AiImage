import torch
from transformers import DALLEModel, DALLETokenizer

# Load the pre-trained DALL-E model and tokenizer
model = DALLEModel.from_pretrained("dalle-large")
tokenizer = DALLETokenizer.from_pretrained("dalle-large")

# Define the text prompt
prompt = "A futuristic cityscape with towering skyscrapers and flying cars"

# Preprocess the text prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate the image
outputs = model.generate(input_ids, num_beams=4, max_length=256)

# Convert the generated image to a PIL image
pil_image = model.feature_extractor.decode(outputs[0])

# Save the generated image to a file
pil_image.save("generated_image.png")
