"""Modal wrapper around the shared credentialed HTTP client."""
from typing import Any

import modal

APP_NAME = "hw2-generation-gateway"
FUNCTION_NAME = "complete"
SECRET_NAME = "hw2-generation-endpoints"
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("requests==2.32.4")
    .add_local_python_source("workflow")
)
endpoint_secret = modal.Secret.from_name(SECRET_NAME)


@app.function(image=image, secrets=[endpoint_secret], timeout=10 * 60)
def complete(request: dict[str, Any]) -> dict[str, Any]:
    from workflow.generation_client import complete as http_complete

    return http_complete(request)
