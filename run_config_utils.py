import os
from enum import Enum
import socket


class Cluster(Enum):
    MIT = {
        "partition": "submit-gpu1080,submit-gpu",
        "root": f"/data/submit/{os.getlogin()}/AI-NUCLEAR-LOGS",
    }
    HARVARD = {
        "partition": "iaifi_gpu",
        "root": "~/data/AI-NUCLEAR-LOGS",
    }


def determine_cluster():
    host = socket.gethostname()

    if host.endswith("mit.edu") or host.startswith("submit"):
        return Cluster.MIT
    elif host.endswith("harvard.edu") or host.startswith("holygpu"):
        return Cluster.HARVARD
    else:
        raise ValueError(f"Unknown cluster: {host}")
