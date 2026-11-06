import time
from datetime import datetime
from .base import BaseLogger

class ExecutionStatsLogger(BaseLogger):
    def log_statistics(self, start_time: float, end_time: float, run_count: int, 
                       batch_update_count: int, completed_plans_count: int, 
                       skillset: dict, brain_skills: dict, skill_stats: dict, 
                       logs: list, continued_from: str = None, 
                       filename: str = "execution_statistics.txt"):
        stats_path = self.get_path(filename)
        if not stats_path:
            return
        
        end_time = time.time() if end_time is None else end_time
        with open(stats_path, 'w') as f:
            if continued_from:
                f.write(f"CONTINUATION FROM SESSION: {continued_from}\n")
                f.write("-" * 100 + "\n")
            
            start_str = datetime.fromtimestamp(start_time).strftime('%d-%m-%Y %H:%M:%S')
            end_str = datetime.fromtimestamp(end_time).strftime('%d-%m-%Y %H:%M:%S')
            duration = end_time - start_time
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            f.write(f"start time: {start_str} | end time: {end_str} | duration: {duration_str}\n")
            f.write(f"runs: {run_count}; brain updates: {batch_update_count}; completed plans: {completed_plans_count}\n")
            f.write("-" * 100 + "\n")
            
            f.write("Collected Data:\n")
            f.write(f"{'Skill Name':<20} | {'precondition (nodes)':<20} | {'effect (nodes)':<20} \n")
            for skill in skillset.keys():
                predicates = brain_skills.get(skill)
                if predicates:
                    f.write(f"{skill:<20} | {len(predicates['precondition'].nodes):<20} | {len(predicates['effect'].nodes):<20} \n")
            
            f.write("-" * 100 + "\n")
            f.write("Executions:\n")
            f.write(f"{'Skill Name':<20} | {'True Positive':<15} | {'True Negative':<15} | {'False Positive':<15} | {'False Negative':<15} | {'Total':<15}\n")
            f.write("-" * 100 + "\n")
            
            for sk_name, counts in skill_stats.items():
                sumall = counts['true_positive'] + counts['true_negative'] + counts['false_positive'] + counts['false_negative']
                f.write(f"{sk_name:<20} | {counts['true_positive']:<15} | {counts['true_negative']:<15} | {counts['false_positive']:<15} | {counts['false_negative']:<15} | {sumall:<10}\n")
            
            f.write("-" * 100 + "\n")
            for log in logs:
                f.write(f"{log}\n")

    def log_timestamps(self, timestamps: list, filename: str = "timestamps.txt"):
        path = self.get_path(filename)
        if not path:
            return
        with open(path, 'w') as f:
            for stmp in timestamps:
                f.write(f"{stmp}\n")
