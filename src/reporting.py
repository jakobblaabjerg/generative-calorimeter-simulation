import os
import numpy as np

class FilterReport:

    def __init__(self, name: str, before: dict, after: dict, params: dict | None = None):

        self.name = name 
        self.params = params or {}
        self.before = before
        self.after = after

    @property
    def events_removed(self):
        return np.setdiff1d(self.before["unique_events"], self.after["unique_events"])

    @property
    def num_steps_removed(self):
        return self.before["num_steps"]-self.after["num_steps"]
                
    @property
    def num_events_removed(self):
        return self.before["num_events"]-self.after["num_events"]
    


class DatasetReport:

    def __init__(self):
        self.reports: list[FilterReport] = []


    def add(self, report: FilterReport):
        self.reports.append(report)


    def write(self, save_dir, file_name):

        file_path = os.path.join(save_dir, f"{file_name}_summary.txt")

        with open(file_path, "w") as f:

            f.write(f"Dataset name: {file_name}\n")
            f.write("="*40+"\n\n")

            for r in self.reports:
                                
                f.write(f"Filter: {r.name}\n")
                f.write(f"  Steps removed:  {r.num_steps_removed}\n")
                f.write(f"  Events removed: {r.num_events_removed}\n")

                if r.params:
                    f.write(f"  Parameters:\n")
                    for param, value in r.params.items():
                        f.write(f"    - {param}: {value}\n")
                f.write("-"*30+"\n")





def get_num_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def model_summary(model):
    
    total_params = 0
    trainable_params = 0

    print("\nModel Summary")
    print("-"*70)
    for name, module in model.named_children():        
        n_params = get_num_params(module)
        n_trainable = get_num_params(module, trainable_only=True)
        total_params += n_params
        trainable_params += n_trainable
        print(f"{name:<20} | params: {n_params:>10} | trainable: {n_trainable:>10}")        
    print("-"*70)
    print(f"{'Total':<20} | params: {total_params:>10} | trainable: {trainable_params:>10}")
    print()

