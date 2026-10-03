"""V3 native adapters use a separate entrypoint from the V1 utility nodes."""
from h3_pipeline.server_nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
