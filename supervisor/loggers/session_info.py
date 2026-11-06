import os
from datetime import datetime
from .base import BaseLogger

class SessionInfoLogger(BaseLogger):
    def log_info(self, config: dict, agent, robot, arena, filename: str = "info.txt"):
        info_file_path = self.get_path(filename)
        brain = getattr(agent, "brain", None)
        
        hyperparams = {
            "precision": getattr(brain, "precision", "?"),
            "n_test_points": getattr(brain, "n_test_points", "?"),
            "min_nodes_in_symbol": getattr(brain, "min_nodes", "?"),
            "span_tolerance": getattr(brain, "span_tolerance", "?"),
            "base_radius": getattr(brain, "base_radius", "?"),
            "max_radius": getattr(brain, "max_radius", "?"),
            "learning_rate_b": getattr(brain, "learning_rate_b", "?"),
            "learning_rate_n": getattr(brain, "learning_rate_n", "?"),
            "max_edge_age": getattr(brain, "max_edge_age", "?"),
            "lambda_step": getattr(brain, "lambda_step", "?"),
            "alpha": getattr(brain, "alpha", "?"),
            "d": getattr(brain, "d", "?"),
        }

        content = [
            "========================================",
            "          EXPERIMENT RUN INFO           ",
            "========================================",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- SESSION CONFIGURATION ---",
            f"Session Directory: {self.log_dir}",
            f"Source Dir (continue/new): {config.get('source_dir')}",
            f"Runs per epoch: {config.get('runs_per_epoch')}",
            f"Tests between epochs: {config.get('tests_between')}",
            f"Tests at the end: {config.get('tests_at_end')}",
            f"Max epochs: {config.get('max_epochs')}",
            f"Time limit: {config.get('time_limit')} hours",
            f"Min success rate: {config.get('min_success_rate')}",
            "",
            "--- COMPONENT TYPES ---",
            f"Agent type: {type(agent).__name__}",
            f"Robot type: {type(robot).__name__}",
            f"Arena type: {type(arena).__name__}",
            "",
            "--- BRAIN NETWORK HYPERPARAMETERS ---",
        ]

        for param, val in hyperparams.items():
            content.append(f"{param}: {val}")

        content.append("========================================\n")

        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        print(f"[Logger] Saved session metadata to {info_file_path}")
