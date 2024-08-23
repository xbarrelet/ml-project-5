import gradio as gr
import requests

AWS_PREDICT_ENDPOINT_URL = "http://inferring-service-env.eba-knxts2bh.eu-west-3.elasticbeanstalk.com/predict"

def predict_tags(title, body):
    json_payload = {
        "body": body,
        "title": title
    }
    json_response = requests.post(AWS_PREDICT_ENDPOINT_URL, json=json_payload).json()
    return json_response.get("predicted_tags")

demo = gr.Interface(
    fn=predict_tags,
    inputs=["text", "text"],
    outputs=["text"],
)

demo.launch()
