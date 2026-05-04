#!/usr/bin/env python3
"""
Ollama inference module for analyzing images using Ollama Cloud API with Vision Language Models.
"""

import os
import base64
import sys
import time
from pathlib import Path
from ollama import Client


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_ollama_client(api_key: str = None) -> Client:
    """
    Get an authenticated Ollama client.

    Args:
        api_key: Ollama API key (optional, will use env var if not provided)

    Returns:
        Ollama Client instance
    """
    if api_key is None:
        api_key = os.environ.get('OLLAMA_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY not provided. "
            "Set it with: export OLLAMA_API_KEY=your_api_key "
            "or pass it as an argument."
        )

    return Client(
        host='https://ollama.com',
        headers={'Authorization': f'Bearer {api_key}'}
    )


def analyze_image_ollama(
    image_path: str,
    prompt: str = "What's in this image?",
    model: str = "qwen3-vl:235b-instruct",
    api_key: str = None,
    verbose: bool = True,
    temperature: float = 0.0
) -> str:
    """
    Analyze an image using Ollama Cloud API with VLM model.

    Args:
        image_path: Path to the image file
        prompt: Question or instruction about the image
        model: Model name (default: qwen3-vl:235b-instruct)
        api_key: Ollama API key (optional)
        verbose: Whether to print progress information
        temperature: Sampling temperature (default: 0.0 for deterministic output)

    Returns:
        Response from the API
    """
    # Validate image path
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get client
    client = get_ollama_client(api_key)

    # Encode the image
    image_base64 = encode_image(image_path)

    if verbose:
        print(f"Analyzing image: {image_path}")
        print(f"Provider: Ollama")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print("-" * 70)

    # Prepare messages with image
    messages = [
        {
            'role': 'user',
            'content': prompt,
            'images': [image_base64]
        }
    ]

    return _chat_with_retries(client, model, messages, verbose, temperature=temperature)


def analyze_multiple_images_ollama(
    image_paths: list,
    prompt: str = "What's in these images?",
    model: str = "qwen3-vl:235b-instruct",
    api_key: str = None,
    verbose: bool = True,
    temperature: float = 0.0
) -> str:
    """
    Analyze multiple images with a single prompt using Ollama Cloud API.

    Args:
        image_paths: List of paths to image files
        prompt: Question or instruction about the images
        model: Model name (default: qwen3-vl:235b-instruct)
        api_key: Ollama API key (optional)
        verbose: Whether to print progress information
        temperature: Sampling temperature (default: 0.0 for deterministic output)

    Returns:
        Response from the API
    """
    # Validate image paths
    for image_path in image_paths:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get client
    client = get_ollama_client(api_key)

    # Encode all images
    images_base64 = [encode_image(img_path) for img_path in image_paths]

    if verbose:
        print(f"Analyzing {len(image_paths)} images:")
        for img_path in image_paths:
            print(f"  - {img_path}")
        print(f"Provider: Ollama")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print("-" * 70)

    # Prepare messages with multiple images
    messages = [
        {
            'role': 'user',
            'content': prompt,
            'images': images_base64
        }
    ]

    return _chat_with_retries(client, model, messages, verbose, temperature=temperature)


def _chat_with_retries(client: Client, model: str, messages: list, verbose: bool, max_retries: int = 5, temperature: float = 0.0) -> str:
    """Send chat request with retries for transient server/network errors."""
    backoff = 1
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            full_response = ""
            if verbose:
                print("\nResponse:")

            for part in client.chat(model, messages=messages, stream=True, options={"temperature": temperature}):
                content = part['message']['content']
                if verbose:
                    print(content, end='', flush=True)
                full_response += content

            if verbose:
                print()  # newline after response

            return full_response

        except Exception as e:
            last_error = e
            status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
            message = str(e).lower()
            retryable = (
                (status_code is not None and 500 <= int(status_code) < 600)
                or "503" in message
                or "server error" in message
                or "temporarily unavailable" in message
                or "connection reset" in message
                or "timeout" in message
            )

            if retryable and attempt < max_retries:
                if verbose:
                    print(f"\nReceived server/network error from Ollama. Retrying in {backoff}s (attempt {attempt}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2
                continue
            break

    raise RuntimeError(f"Error making request to Ollama: {last_error}")
