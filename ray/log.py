import ray
from ray.experimental.state.api import list_nodes, list_workers, list_tasks
from ray._private import state
import json
import os

def save_ray_state():
    os.makedirs("/volume/data/tldu/new/ai4s-job-system/submodules/verl/ray", exist_ok=True)

    nodes = list_nodes()
    breakpoint()
    with open("/volume/data/tldu/new/ai4s-job-system/submodules/verl/ray/nodes.json", "w") as f:
        json.dump(nodes, f, indent=2)

    workers = list_workers()
    with open("/volume/data/tldu/new/ai4s-job-system/submodules/verl/ray/workers.json", "w") as f:
        json.dump(workers, f, indent=2)

    tasks = list_tasks()
    with open("/volume/data/tldu/new/ai4s-job-system/submodules/verl/ray/tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)

    print("Ray dashboard 状态已保存")
save_ray_state()