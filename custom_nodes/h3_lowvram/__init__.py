"""Install this directory into ComfyUI/custom_nodes after installing the package."""
from h3_pipeline.server_nodes import NODE_CLASS_MAPPINGS as BASE_NODES
from h3_pipeline.dance_nodes import NODE_CLASS_MAPPINGS as DANCE_NODES
from h3_pipeline.dance_reports import install_execution_reporting

install_execution_reporting()

NODE_CLASS_MAPPINGS = {**BASE_NODES, **DANCE_NODES}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
