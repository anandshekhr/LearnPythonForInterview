from .base import JobSourceAdapter
from .greenhouse import GreenhouseAdapter
from .lever import LeverAdapter

ADAPTERS: dict[str, type[JobSourceAdapter]] = {
    GreenhouseAdapter.source_type: GreenhouseAdapter,
    LeverAdapter.source_type: LeverAdapter,
}


def get_adapter_class(source_type: str) -> type[JobSourceAdapter]:
    try:
        return ADAPTERS[source_type]
    except KeyError as exc:
        raise ValueError(f"No adapter registered for source type '{source_type}'") from exc


def adapter_for(source, http=None, with_credentials=False) -> JobSourceAdapter:
    from ..http import SourceHttpClient

    cls = get_adapter_class(source.source_type)
    http = http or SourceHttpClient(throttle_key=f"{source.source_type}:{source.identifier}",
                                    min_interval=source.min_request_interval_seconds)
    api_key = source.get_submission_api_key() if with_credentials and source.has_submission_key else ""
    return cls(source.identifier, source.config, http=http, api_key=api_key)
