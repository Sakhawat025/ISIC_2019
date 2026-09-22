import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "fl_checkpoints"
NUM_ROUNDS = 5


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Keep each run in a separate folder.
        self.run_dir = CHECKPOINT_DIR / uuid4().hex
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Record which run the notebook should evaluate.
        manifest = {
            "run_dir": str(self.run_dir),
            "num_rounds": NUM_ROUNDS,
        }

        manifest_path = CHECKPOINT_DIR / "latest_run.json"
        temp_path = manifest_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, manifest_path)

    def aggregate_fit(self, server_round, results, failures):
        # Require all four clients for a complete round.
        if failures or len(results) != 4:
            raise RuntimeError(
                f"Round {server_round}: "
                f"{len(results)} results, {len(failures)} failures."
            )

        parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if parameters is None:
            raise RuntimeError(
                f"No aggregated weights for round {server_round}."
            )

        arrays = parameters_to_ndarrays(parameters)

        checkpoint_path = (
            self.run_dir / f"round_{server_round}.npz"
        )
        temp_path = checkpoint_path.with_suffix(".tmp")

        # Write completely before exposing the checkpoint.
        with temp_path.open("wb") as file:
            np.savez(
                file,
                **{
                    f"param_{i:05d}": array
                    for i, array in enumerate(arrays)
                },
            )

        os.replace(temp_path, checkpoint_path)

        print(
            f"Round {server_round}: global weights saved to "
            f"{checkpoint_path}",
            flush=True,
        )

        return parameters, metrics


def get_strategy():
    return SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=4,
        min_available_clients=4,
        accept_failures=False,
    )


def start_server():
    return fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS,
        ),
        strategy=get_strategy(),
    )


if __name__ == "__main__":
    print("Starting Flower Server...", flush=True)
    start_server()