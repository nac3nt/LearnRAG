import importlib

import config
from src.embeddings.image_base import BaseImageEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_image_embedder() -> BaseImageEmbedder | None:
    """
    Load the configured image embedder, or return None when disabled.

    The `custom` provider allows the user to add a local class later and
    point `IMAGE_EMBEDDER_CLASS` at it without changing the ingestion code.
    """
    if config.IMAGE_EMBED_PROVIDER == "none":
        return None

    if config.IMAGE_EMBED_PROVIDER == "custom":
        return _load_custom_embedder(config.IMAGE_EMBEDDER_CLASS)

    logger.warning(
        f"Unknown image embed provider '{config.IMAGE_EMBED_PROVIDER}'. "
        "Image vectors will be skipped."
    )
    return None


def _load_custom_embedder(dotted_path: str) -> BaseImageEmbedder:
    """Import and instantiate a custom image embedder class."""
    if "." not in dotted_path:
        raise ValueError(
            "IMAGE_EMBEDDER_CLASS must be a dotted import path like "
            "'src.embeddings.my_embedder.MyImageEmbedder'"
        )

    module_name, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    embedder = cls()

    if not isinstance(embedder, BaseImageEmbedder):
        raise TypeError(
            "Custom image embedder must inherit BaseImageEmbedder. "
            f"Got {type(embedder)!r} from {dotted_path}."
        )

    return embedder
