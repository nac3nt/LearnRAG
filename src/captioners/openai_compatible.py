import base64
import time

import requests

import config
from src.captioners.base import BaseImageCaptioner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAICompatibleVisionCaptioner(BaseImageCaptioner):
    """
    Image captioner using an OpenAI-compatible chat completions endpoint.

    This keeps the PDF ingestion pipeline provider-agnostic: any endpoint
    that supports text+image chat payloads can be configured via .env.
    """

    def __init__(self) -> None:
        self._api_key = config.VISION_API_KEY
        self._base_url = config.VISION_BASE_URL.rstrip("/")
        self._model = config.VISION_MODEL
        self._endpoint = f"{self._base_url}/v1/chat/completions"

        missing = [
            name
            for name, value in {
                "VISION_API_KEY": self._api_key,
                "VISION_BASE_URL": self._base_url,
                "VISION_MODEL": self._model,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "OpenAI-compatible vision captioning requires: "
                + ", ".join(missing)
            )

    def name(self) -> str:
        return f"openai-compatible/{self._model}"

    def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        filename: str,
        page_number: int,
        image_index: int,
    ) -> str:
        if not image_bytes:
            raise ValueError("image_bytes must not be empty")

        data_url = _to_data_url(image_bytes, mime_type)
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": config.VISION_DESCRIPTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Source PDF: "
                                f"{filename}, page {page_number}, image {image_index + 1}. "
                                "Return plain text only."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        }

        return self._call_api(payload)

    def _call_api(self, payload: dict) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        backoff_seconds = config.VISION_INITIAL_BACKOFF_SECONDS
        max_attempts = config.VISION_MAX_RETRIES + 1

        for attempt in range(1, max_attempts + 1):
            response: requests.Response | None = None

            try:
                response = requests.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=config.VISION_REQUEST_TIMEOUT_SEC,
                )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]["content"]
                text = _coerce_content_to_text(message).strip()
                if not text:
                    raise RuntimeError("Vision model returned an empty description.")
                return text
            except requests.exceptions.Timeout as exc:
                error_message = (
                    "Vision API request timed out after "
                    f"{config.VISION_REQUEST_TIMEOUT_SEC:g}s."
                )
                should_retry = True
                cause = exc
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else "unknown"
                response_text = exc.response.text[:300] if exc.response else str(exc)
                error_message = (
                    f"Vision API HTTP error {status_code}: {response_text}"
                )
                should_retry = bool(
                    exc.response
                    and exc.response.status_code
                    in config.VISION_RETRYABLE_STATUS_CODES
                )
                cause = exc
            except requests.exceptions.RequestException as exc:
                error_message = f"Vision API connection error: {exc}"
                should_retry = True
                cause = exc
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                response_text = response.text[:300] if response is not None else ""
                raise RuntimeError(
                    f"Unexpected vision API response shape: {exc}\n"
                    f"Response: {response_text}"
                ) from exc

            if attempt == max_attempts or not should_retry:
                raise RuntimeError(error_message) from cause

            logger.warning(
                f"Vision API call failed on attempt {attempt}/{max_attempts}: "
                f"{error_message} Retrying in {backoff_seconds:.1f}s."
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

        raise RuntimeError("Vision API request failed unexpectedly.")


def _coerce_content_to_text(content: str | list[dict]) -> str:
    """Normalize provider-specific message content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(part for part in text_parts if part)

    raise TypeError(f"Unsupported message content type: {type(content)!r}")


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
