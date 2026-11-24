from pathlib import Path
from supervisor.core.picture_generator import PictureGenerator
from supervisor.generators.exec_stats import ExecStatsGenerator
from supervisor.generators.histogram import HistogramPlotGenerator, UpdateFrequencyPictureGenerator, LogModeDistributionGenerator
from supervisor.utils.dir_utils import get_latest_session
from supervisor.generators.symbols_repr import SymbolIconPictureGenerator, SymbolCategoryPictureGenerator


class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root='logs', session_dir=None):
        if session_dir is None:
            latest = get_latest_session(logs_root)
            if latest:
                session_dir = str(latest)

        super().__init__(logs_root, session_dir)
        self.generators = [
            UpdateFrequencyPictureGenerator(logs_root, session_dir, bin_size_sec=600.0),
            HistogramPlotGenerator(logs_root, session_dir, bw_method=300.0),
            LogModeDistributionGenerator(logs_root, session_dir, bin_size=100),
            ExecStatsGenerator(logs_root, session_dir),
            SymbolIconPictureGenerator(logs_root, session_dir),
            SymbolCategoryPictureGenerator(logs_root, session_dir),
        ]

    def run(self):
        for gen in self.generators:
            gen.run()

if __name__ == "__main__":
    session = None #r"C:\Users\klara\source\repos\LicencjatKogni\logs\2026-11-20_09-09"
    gen = PipelinePictureGenerator(session_dir=session)
    gen.run()