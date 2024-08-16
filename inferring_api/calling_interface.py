from pprint import pprint

import requests

AWS_BEANSTALK_URL = "http://inferring-service-env.eba-knxts2bh.eu-west-3.elasticbeanstalk.com"
PREDICT_ENDPOINT = "/predict"

TEST_PAYLOAD_1 = {
    "body": "<p>How do I find all files containing a specific string of text within their file contents?</p>\n<p>The following doesn't work. It seems to display every single file in the system.</p>\n<pre class=\"lang-none prettyprint-override\"><code>find / -type f -exec grep -H 'text-to-find-here' {} \\;\n</code></pre>\n",
    "title": "Find all files containing a specific text (string) on Linux?"
}

if __name__ == '__main__':
    print("")
    print("Calling inferring service with test payload 1:\n")

    json_response = requests.post(AWS_BEANSTALK_URL + PREDICT_ENDPOINT, json=TEST_PAYLOAD_1).json()

    predicted_tags = json_response.get("predicted_tags")
    print(f"Predicted tags for test payload 1:{predicted_tags}")