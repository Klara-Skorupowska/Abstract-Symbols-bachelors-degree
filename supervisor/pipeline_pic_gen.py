from supervisor.picture_generator import PictureGenerator
from supervisor.symbol_pic_gen import SymbolIconPictureGenerator, SymbolCategoryPictureGenerator
from supervisor.trans_graph_pic_gen import TransGraphPictureGenerator
from supervisor.update_freq_pic_gen import UpdateFrequencyPictureGenerator

class PipelinePictureGenerator(PictureGenerator):
    def __init__(self, logs_root="logs", session_dir=None):
        super().__init__(logs_root=logs_root, session_dir=session_dir)
        self.generators = [
            SymbolIconPictureGenerator(logs_root, session_dir=session_dir),
            SymbolCategoryPictureGenerator(logs_root, session_dir=session_dir),
            TransGraphPictureGenerator(logs_root, session_dir=session_dir),
            UpdateFrequencyPictureGenerator(logs_root, session_dir=session_dir, bin_size_sec=600.0)
        ]

    def run(self):
        for gen in self.generators:
            gen.run()

if __name__ == "__main__":
    sd = None
    gen = PipelinePictureGenerator(session_dir=sd)
    gen.run()