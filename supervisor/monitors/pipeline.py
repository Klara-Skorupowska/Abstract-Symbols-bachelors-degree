from supervisor.core.live_monitor import LiveMonitor
from supervisor.monitors.symbols import SymbolsLive
from supervisor.monitors.sensimotor import LiveSensimotorMonitor

class PipelineLiveMonitor(LiveMonitor):
    def setup(self):
        self.monitors = [
            SymbolsLive(self.bus, 'Current Symbols'), 
            LiveSensimotorMonitor(self.bus, 'The Eyes')
            ]
        for mon in self.monitors: mon.setup()

    def update(self):
        for mon in self.monitors: mon.update()

    def close(self):
        for mon in self.monitors: mon.close()
