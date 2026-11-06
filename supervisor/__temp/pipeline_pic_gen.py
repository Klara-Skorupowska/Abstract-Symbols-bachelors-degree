from supervisor.picture_generator import PictureGenerator
from supervisor.symbol_pic_gen import SymbolIconPictureGenerator, SymbolCategoryPictureGenerator
from supervisor.trans_graph_pic_gen import TransGraphPictureGenerator
from supervisor.update_freq_pic_gen import UpdateFrequencyPictureGenerator
from supervisor.update_freq_cumulative_pic_gen import HistogramPlotGenerator
from supervisor.exec_stats_cumulative_pic_gen import ExecStatsGenerator
from supervisor.symbol_stats_cumulative_pic_gen import SymbolStatsGenerator

class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)
        self.generators = [
            SymbolIconPictureGenerator(logs_root, session_dir=session_dir),
            SymbolCategoryPictureGenerator(logs_root, session_dir=session_dir),
            SymbolStatsGenerator(logs_root, session_dir=session_dir),
            UpdateFrequencyPictureGenerator(logs_root, session_dir=session_dir, bin_size_sec=10.0),
            HistogramPlotGenerator(logs_root, session_dir=session_dir),
            ExecStatsGenerator(logs_root, session_dir=session_dir),

        ]

    def run(self):
        for gen in self.generators:
            gen.run()

if __name__ == "__main__":
    sd = None
    gen = PipelinePictureGenerator(session_dir=sd)
    gen.run()