from torch_geometric.nn import MessagePassing


class Double_Level_MessageProp_random_walk_wo_norm(MessagePassing):
    def __init__(self, node_dim=-3):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, qv_gate):
        out = self.propagate(edge_index, x=x, qv_gate=qv_gate)
        return out

    def message(self, x_j, qv_gate):
        return qv_gate * x_j


class Double_Level_KeyProp_random_walk_wo_norm(MessagePassing):
    def __init__(self, node_dim=-2):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, qv_gate):
        out = self.propagate(edge_index, x=x, qv_gate=qv_gate)
        return out

    def message(self, x_j, qv_gate):
        return qv_gate * x_j


class Double_Level_MessageProp_random_walk_w_norm(MessagePassing):
    def __init__(self, node_dim=-3):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, norm, qv_gate):
        out = self.propagate(edge_index, x=x, norm=norm, qv_gate=qv_gate)
        return out

    def message(self, x_j, norm, qv_gate):
        return qv_gate * norm * x_j


class Double_Level_KeyProp_random_walk_w_norm(MessagePassing):
    def __init__(self, node_dim=-2):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, norm, qv_gate):
        out = self.propagate(edge_index, x=x, norm=norm, qv_gate=qv_gate)
        return out

    def message(self, x_j, norm, qv_gate):
        return qv_gate * norm * x_j


class MessageProp_random_walk(MessagePassing):
    def __init__(self, node_dim=-3):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm * x_j


class KeyProp_random_walk(MessagePassing):
    def __init__(self, node_dim=-2):
        super().__init__(aggr='add', node_dim=node_dim)

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm * x_j
