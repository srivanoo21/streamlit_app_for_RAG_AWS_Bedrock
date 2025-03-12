import os
import json
import sys
import boto3


prompt="""
        you are a cricket expert now just tell me when RCB will win the IPL?
"""   
 
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

payload={
    "prompt": "[INST]"+prompt+"[/INST]",
    "max_gen_len": 512,
    "temperature": 0.3,
    "top_p":0.9
}

body = json.dumps(payload) 
model_id = "meta.llama3-8b-instruct-v1:0" 

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse response
response_body = json.loads(response.get("body").read())
print("Raw Response:", response_body)

# Extract and print the actual response
# response_text = response_body.get("generation", "No response generated")
# print("Response:", response_text)

text_response = response_body["generation"].strip().split('\n')[0]  # This assumes the text you need is the first line
print("Response:", text_response)