from .base_frame import BaseFrame


class BaseNetworkFrame(BaseFrame):
    def __init__(self, network, source_nodes=None, target_nodes=None) -> None:
        super().__init__(network, source_nodes, target_nodes)

    @property
    def source_nodes(self):
        return self.row_objects

    @property
    def target_nodes(self):
        return self.col_objects

    @property
    def nodes(self):
        raise NotImplementedError()
        # TODO something about checking whether source/target are the same thing?
        pass
