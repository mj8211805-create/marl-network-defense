"""Enterprise network topology model for CyberMARL simulation."""

import random
from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field


class NodeType(str, Enum):
    GATEWAY = "Gateway"
    DMZ_SERVER = "DMZ_Server"
    INTERNAL_SERVER = "Internal_Server"
    DATABASE = "Database"
    WORKSTATION = "Workstation"
    DOMAIN_CONTROLLER = "Domain_Controller"


class NodeStatus(str, Enum):
    HEALTHY = "Healthy"
    DEGRADED = "Degraded"
    COMPROMISED = "Compromised"
    ISOLATED = "Isolated"


@dataclass
class NetworkNode:
    node_id: str
    name: str
    node_type: NodeType
    ip_address: str
    subnet: str
    status: NodeStatus = NodeStatus.HEALTHY
    cpu_load: float = 0.1  # 0.0 to 1.0
    bandwidth_usage_mbps: float = 5.0
    compromise_level: float = 0.0  # 0.0 (clean) to 1.0 (fully compromised)
    active_connections: int = 10
    auth_failures: int = 0
    is_protected_by_agent: bool = True

    def reset(self):
        self.status = NodeStatus.HEALTHY
        self.cpu_load = random.uniform(0.05, 0.20)
        self.bandwidth_usage_mbps = random.uniform(2.0, 10.0)
        self.compromise_level = 0.0
        self.active_connections = random.randint(5, 20)
        self.auth_failures = 0


class NetworkTopology:
    """Manages the graph of nodes, subnets, and routing topology."""

    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self._build_enterprise_topology()

    def _build_enterprise_topology(self):
        # 1. Gateway
        self.nodes["gw-01"] = NetworkNode("gw-01", "Perimeter Gateway Router", NodeType.GATEWAY, "192.168.1.1", "192.168.1.0/24")

        # 2. DMZ Web & App Servers
        self.nodes["dmz-web-01"] = NetworkNode("dmz-web-01", "Public Web Portal", NodeType.DMZ_SERVER, "10.0.1.10", "10.0.1.0/24")
        self.nodes["dmz-api-01"] = NetworkNode("dmz-api-01", "REST API Gateway", NodeType.DMZ_SERVER, "10.0.1.20", "10.0.1.0/24")

        # 3. Internal Application & File Servers
        self.nodes["srv-app-01"] = NetworkNode("srv-app-01", "Enterprise ERP Server", NodeType.INTERNAL_SERVER, "10.0.2.10", "10.0.2.0/24")
        self.nodes["srv-file-01"] = NetworkNode("srv-file-01", "Corporate File Server", NodeType.INTERNAL_SERVER, "10.0.2.20", "10.0.2.0/24")

        # 4. Critical Database Cluster
        self.nodes["db-primary-01"] = NetworkNode("db-primary-01", "Core SQL Database", NodeType.DATABASE, "10.0.3.10", "10.0.3.0/24")
        self.nodes["dc-primary-01"] = NetworkNode("dc-primary-01", "Active Directory DC", NodeType.DOMAIN_CONTROLLER, "10.0.3.20", "10.0.3.0/24")

        # 5. Workstations Subnet
        for i in range(1, 5):
            w_id = f"ws-corp-{i:02d}"
            self.nodes[w_id] = NetworkNode(w_id, f"Workstation User {i}", NodeType.WORKSTATION, f"10.0.4.{10+i}", "10.0.4.0/24")

    def reset(self):
        for node in self.nodes.values():
            node.reset()

    def get_overall_health_ratio(self) -> float:
        """Returns percentage of healthy nodes."""
        healthy = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
        return healthy / len(self.nodes)

    def get_average_cpu_load(self) -> float:
        return sum(n.cpu_load for n in self.nodes.values()) / len(self.nodes)

    def get_compromised_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status == NodeStatus.COMPROMISED)

    def to_graph_data(self) -> Dict[str, Any]:
        """Serializes topology for web visualization (Vis.js compatible)."""
        nodes_list = []
        for n in self.nodes.values():
            color = "#00ff9d" if n.status == NodeStatus.HEALTHY else ("#ff0055" if n.status == NodeStatus.COMPROMISED else "#fbbf24")
            nodes_list.append({
                "id": n.node_id,
                "label": f"{n.name}\n({n.ip_address})",
                "type": n.node_type.value,
                "status": n.status.value,
                "cpu": f"{n.cpu_load*100:.0f}%",
                "color": color
            })
        
        edges_list = [
            {"from": "gw-01", "to": "dmz-web-01"},
            {"from": "gw-01", "to": "dmz-api-01"},
            {"from": "dmz-web-01", "to": "srv-app-01"},
            {"from": "srv-app-01", "to": "db-primary-01"},
            {"from": "srv-app-01", "to": "dc-primary-01"},
            {"from": "gw-01", "to": "ws-corp-01"},
            {"from": "gw-01", "to": "ws-corp-02"},
            {"from": "ws-corp-01", "to": "srv-file-01"},
            {"from": "ws-corp-02", "to": "srv-file-01"},
        ]
        return {"nodes": nodes_list, "edges": edges_list}
