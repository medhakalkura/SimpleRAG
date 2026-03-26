"""
Step 3: Image generation via NVIDIA API (Stable Diffusion 3 Medium).
Requires NVIDIA_API_KEY. Get one at https://build.nvidia.com/
"""

import os
import base64
import tempfile
import requests


def generate_image(prompt: str, size: str = "1024x1024", model: str = "dall-e-3") -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Get a key at https://build.nvidia.com/"
        )

    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt[:4000],
        "cfg_scale": 5,
        "aspect_ratio": "16:9",
        "seed": 0,
        "steps": 50,
        "negative_prompt": "",
    }

    response = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    response_body = response.json()
    image_b64 = response_body.get("image")
    if not image_b64:
        raise RuntimeError("No image in NVIDIA API response")

    image_bytes = base64.b64decode(image_b64)

    path = os.path.join(tempfile.gettempdir(), "rag_generated_image.jpg")
    with open(path, "wb") as f:
        f.write(image_bytes)

    return path