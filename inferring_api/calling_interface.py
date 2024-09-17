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

# For the demo
test_payload = {
    "title": "Does Python have a string &#39;contains&#39; substring method?",
    "body": "<p>I'm looking for a <code>string.contains</code> or <code>string.indexof</code> method in Python.</p>\n\n<p>I want to do:</p>\n\n<pre><code>if not somestring.contains(\"blah\"):\n   continue\n</code></pre>\n"
}