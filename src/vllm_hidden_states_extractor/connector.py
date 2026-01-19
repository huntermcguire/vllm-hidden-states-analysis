from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)


class ModifiedExampleConnector(ExampleConnector):
    def clear_connector_metadata(self) -> None:
        pass

    def real_clear_connector_metadata(self):
        self._connector_metadata = None
