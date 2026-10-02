"""No-download synthetic demonstration of both classifier paths."""
import json
import numpy as np
import pandas as pd
import torch
from star_classifier import run_experiment


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(90, 10)), columns=[f"feature_{i}" for i in range(10)])
    frame["class"] = np.arange(90) % 3
    result = run_experiment(frame, epochs=2, seed=42)
    print(json.dumps({"scope": "synthetic execution check; not stellar classification quality",
                      "models": {name: {"test_samples": metrics["samples"], "epochs": len(result["history"][name])}
                                 for name, metrics in result["metrics"].items()}}))


if __name__ == "__main__":
    main()
