import boto3  # AWS SDK to interact with AWS services like Bedrock
import json   # For handling JSON data
import base64 # Encoding and decoding Base64 data
import os     # Operating system functionalities like creating directories and handling files


# Prompt preparation:
# prompt: Text input describing the desired image.
# prompt_template: List containing dictionaries where "text" holds the prompt, and "weight" defines 
# how much emphasis should be given to this prompt during image generation.

prompt="""
provide me one 4k hd image of person who is standing over the mount everest peak.
"""
prompt_template=[{"text":prompt,"weight":1}]


# This creates a Bedrock client to interact with AWS Bedrock service in the us-east-1 region.
bedrock=boto3.client(service_name="bedrock-runtime", region_name="us-east-1")


# Payload preparation:
# text_prompts: List of prompts for the model.
# cfg_scale: Classifier-Free Guidance scale (higher value means the model will follow the prompt more strictly).
# seed: Random seed to ensure reproducibility.
# steps: Number of inference steps (more steps = better quality).
# width and height: Image dimensions (512x512 pixels).

payload={ 
    "text_prompts":prompt_template,
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":512,
    "height":512
    
}

# converting payload to json
body=json.dumps(payload)
model_id="stability.stable-diffusion-xl-v1"


# Invole AWS bedrock model:

# This line sends the request to AWS Bedrock with:
# body: Payload data.
# modelId: Model identifier.
# accept: Expected response format.
# contentType: Content format sent to the model.

response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)


# Extracting Response
response_body=json.loads(response.get("body").read())
print(response_body)


# Base64 Image Decoding:
# artifacts: List containing generated images.
# base64: Retrieves the Base64-encoded image string.
# Base64 decoding converts the encoded image back into bytes.

artifacts = response_body.get("artifacts")[0]
image_encoded = artifacts.get("base64").encode('utf-8')
image_bytes = base64.b64decode(image_encoded)


# Saving Image Locally
output_dir="output"
os.makedirs(output_dir,exist_ok=True)
file_name=f"{output_dir}/generated-img.png"
with open(file_name,"wb") as f:
    f.write(image_bytes)