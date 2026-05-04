"""Label Studio SDK client wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ls_cli.config import Settings

if TYPE_CHECKING:
    from label_studio_sdk.client import LabelStudio


def get_client(settings: Settings) -> LabelStudio:
    from label_studio_sdk.client import LabelStudio

    return LabelStudio(base_url=settings.label_studio_url, api_key=settings.label_studio_api_key)


def auth_headers(client: LabelStudio) -> dict[str, str]:
    """Auth headers for endpoints not covered by SDK methods (image presign, downloads)."""
    return client._client_wrapper.get_headers()
