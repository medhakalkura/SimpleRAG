"""
Step 3: Image generation from a text prompt.

Anthropic's API does not generate images; it does vision (understanding images).
So we use OpenAI's DALL·E 3 (or you can plug in Replicate/Stability/etc.) to generate
the image from the enhanced prompt. The "Claude + RAG" part improves the prompt;
this module turns that prompt into an image.

Requires OPENAI_API_KEY for DALL·E.
"""

import os
import base64
import urllib.request
import tempfile

from openai import OpenAI


def generate_image(prompt: str, size: str = "1024x1024", model: str = "dall-e-3") -> str:
    """
    Call OpenAI Images API (DALL·E 3) to generate an image from the given prompt.
    Returns a local file path where the image was saved (or a data URL if you prefer).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it to use DALL·E image generation.")

    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model=model,
        prompt=prompt[:4000],  # API limit
        size=size,
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    if not image_url:
        raise RuntimeError("No image URL in API response.")

    # Download to a temp file and return path (so we can display or save)
    path = os.path.join(tempfile.gettempdir(), "rag_generated_image.png")
    urllib.request.urlretrieve(image_url, path)
    return path
