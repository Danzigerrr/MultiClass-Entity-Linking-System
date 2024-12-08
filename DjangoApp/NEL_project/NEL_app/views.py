import requests
from django.shortcuts import render
import time
import os
from dotenv import load_dotenv
from django.http import JsonResponse


# Load environment variables from .env file
load_dotenv()

# Access the Hugging Face API token
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


def infer_with_hf_api(text, model_name="flair/ner-english-ontonotes", max_retries=10, wait_time=3):
    """
    Use Hugging Face API to infer NER tags with retry mechanism.
    Retries up to `max_retries` times if the model is loading.
    """
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": HF_API_TOKEN}
    payload = {"inputs": text}

    for attempt in range(1, max_retries + 1):
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            # Success: return the response
            return response.json()

        # Parse response error if available
        error_data = response.json()
        if "error" in error_data and "currently loading" in error_data["error"]:
            estimated_time = error_data.get("estimated_time", wait_time)
            print(f"Attempt {attempt}/{max_retries}: Model is loading. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            # Non-retryable error or unexpected error format
            print(f"Non-retryable error: {error_data}")
            return {"error": error_data}

    # If retries exceeded
    return {"error": f"Exceeded maximum retries ({max_retries}). Model might still be loading."}


def index(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Handle AJAX request
        user_input = request.POST.get("user_input", "")
        try:
            result = infer_with_hf_api(user_input)  # Call the Hugging Face API
            output_html = generate_html(result, user_input)  # Generate HTML for tagged entities
            return JsonResponse({"output_html": output_html})  # Return JSON response
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    # For initial page load (non-AJAX requests)
    return render(request, "NEL_app/index.html", {})


def generate_html(ner_results, text):
    """
    Generate HTML output with highlighted NER tags.
    :param ner_results: List of NER result dictionaries from the Hugging Face API.
    :param text: Original input text.
    :return: HTML string with entities highlighted.
    """
    html_str = "<p>"
    start_idx = 0

    # Iterate over entities
    for entity in ner_results:
        entity_start = entity["start"]
        entity_end = entity["end"]
        entity_tag = entity["entity_group"]
        entity_text = text[entity_start:entity_end]

        # Add text before the entity
        html_str += text[start_idx:entity_start]

        # Add highlighted entity
        html_str += f"<span class=\"{entity_tag}\" style=\"background-color: white;\">{entity_text} ({entity_tag})</span>"

        # Update the start index
        start_idx = entity_end

    # Add remaining text
    html_str += text[start_idx:] + "</p>"

    # Add CSS styles for different entity classes (optional, for additional styling)
    css_styles = """
    <style>
        .CARDINAL { color: blue; }
        .DATE { color: green; }
        .EVENT { color: red; }
        .FAC { color: orange; }
        .GPE { color: purple; }
        .LANGUAGE { color: brown; }
        .LAW { color: pink; }
        .LOC { color: gray; }
        .MONEY { color: yellow; }
        .NORP { color: cyan; }
        .ORDINAL { color: olive; }
        .ORG { color: teal; }
        .PERCENT { color: navy; }
        .PERSON { color: maroon; }
        .PRODUCT { color: lime; }
        .QUANTITY { color: gold; }
        .TIME { color: indigo; }
        .WORK_OF_ART { color: violet; }
    </style>
    """

    return html_str + css_styles
