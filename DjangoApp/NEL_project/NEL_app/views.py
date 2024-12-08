import requests
from django.shortcuts import render
import time
import os
from dotenv import load_dotenv
from django.http import JsonResponse
from flair.data import Sentence
from flair.models import SequenceTagger

# Load the Flair NER model once (using the 'fast' version)
print("Loading model...")
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
print("Model loaded.")

# Global sentence variable
sentence = None


def extract_entity_probabilities(entity):
    entity_probabilities = {}

    for token in entity:
        token_probabilities = token.get_tags_proba_dist("ner")
        for token_prob in token_probabilities:
            # Skip "O" class (non-entity tokens)
            if token_prob.value == 'O':
                label = "O"
            else:
                label = token_prob.value[2:]  # Remove the prefix (e.g., B-, I-, E-)
            score = token_prob.score
            entity_probabilities[label] = entity_probabilities.get(label, 0) + score / len(entity)

    # Sort probabilities by score in descending order
    sorted_probabilities = sorted(entity_probabilities.items(), key=lambda x: x[1], reverse=True)

    return sorted_probabilities[:3]


def get_entities_and_probabilities(sentence):
    """
    Extract entities and their top 3 class probabilities from a sentence.
    :param sentence: A Flair Sentence object containing the text.
    :return: A list of dictionaries containing entity text, start/end positions,
             entity group, and their top 3 probabilities.
    """
    ner_results = []

    # Iterate through the entities in the sentence
    for entity in sentence.get_spans('ner'):
        entity_probabilities = extract_entity_probabilities(entity)

        top_3_probabilities = "<ul>"
        for i, (label, probability) in enumerate(entity_probabilities):
            top_3_probabilities += f"<li>{label}: {probability:.4f}</li>"
        top_3_probabilities += "</ul>"

        ner_results.append({
            "text": entity.text,
            "start": entity.start_position,
            "end": entity.end_position,
            "entity_group": entity.get_label("ner").value,
            "probabilities": top_3_probabilities
        })

    return ner_results


def generate_html(ner_results, text):
    """
    Generate HTML output with highlighted NER tags and top 3 class probabilities.
    """
    html_str = "<p>"
    start_idx = 0

    # Iterate over entities
    for entity in ner_results:
        entity_start = entity.get("start", 0)
        entity_end = entity.get("end", 0)
        entity_tag = entity.get("entity_group", "UNKNOWN")
        entity_text = text[entity_start:entity_end]

        # Add text before the entity
        html_str += text[start_idx:entity_start]

        # Add highlighted entity and a clickable link with additional details in a tooltip
        html_str += f'<span class="entity" onclick="showEntityDetails(\'{entity_text}\', \'{entity_tag}\', \'{entity["probabilities"]}\')" style="background-color: yellow; cursor: pointer;">{entity_text} ({entity_tag})</span>'

        # Update the start index
        start_idx = entity_end

    # Add remaining text
    html_str += text[start_idx:] + "</p>"

    # Add CSS styles for different entity classes (optional)
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


def index(request):
    global sentence  # Use the global sentence variable

    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # Handle AJAX request
        user_input = request.POST.get("user_input", "")

        if not user_input:
            return JsonResponse({"error": "Missing 'user_input' parameter"}, status=400)

        try:
            # Process the input using Flair's Sentence object
            sentence = Sentence(user_input)

            # Run NER on the sentence
            tagger.predict(sentence, return_probabilities_for_all_classes=True)

            # Extract entities and their probabilities
            ner_results = get_entities_and_probabilities(sentence)

            output_html = generate_html(ner_results, user_input)  # Generate HTML for tagged entities
            return JsonResponse({"output_html": output_html})  # Return JSON response
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    # For initial page load (non-AJAX requests)
    return render(request, "NEL_app/index.html", {})
