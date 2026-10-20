from .live_monitor import LiveMonitor
from .trans_graph_live import TransGraphLive
from .sensimotor_live import LiveSensimotorMonitor
from .symbols_live import SymbolsLive

class PipelineLiveMonitor(LiveMonitor):
    def setup(self):
        self.monitors = [
                TransGraphLive(self.bus, "Transition Graph"),
                #LiveSensimotorMonitor(self.bus, "Sensimotor Monitor"),
                SymbolsLive(self.bus, "Current Symbols")
            ]
        for mon in self.monitors:
            mon.setup()

    def update(self):
        for mon in self.monitors:
            mon.update()

    def close(self):
        for mon in self.monitors:
            mon.close()