import numpy as np

class BinDetails:
    def __init__(self, bins, free_bin_caps):
        self.bins = bins
        self.free_bin_caps = free_bin_caps
        self.no_of_bins = len(bins)
        self.score = -1
    
    def __repr__(self):
        return f"BinDetails (Score = {self.score}, No_of_bins = {self.no_of_bins})"
    
    def __str__(self):
        return f"BinDetails (Score = {self.score}, No_of_bins = {self.no_of_bins})"

class BinPackingSolutions:
    def __init__(self):
        self.bin_solutions: dict[int, BinDetails] = {}
        self.max_bin_capacity = 30
        self.items = np.array([5, 14, 11, 8, 3, 2, 13, 1, 5, 6, 9, 4, 7, 12, 10, 15])
        self.no_of_items = self.items.size
        self.rng = np.random.default_rng(1234)

    def __repr__(self):
        return f"BinPackingSolutions (No_of_items = {self.no_of_items}, Max_bin_capacity = {self.max_bin_capacity})"
    
    def __str__(self):
        return f"BinPackingSolutions (No_of_items = {self.no_of_items}, Max_bin_capacity = {self.max_bin_capacity})"

    def extract_from_file(self, file_path):
        with open(file_path, "r") as f: 
            lines = f.readlines()
            n = int(lines[0])
            capacity = int(lines[1])
            items = [int(lines[i]) for i in range(2, n+2)]
        
        self.items = np.array(items)
        self.no_of_items = n
        self.max_bin_capacity = capacity

    def best_fit_heuristic(self, id, gen=False):
        free_bin_caps = [self.max_bin_capacity]
        bins = [[]]
        shuffled_items = self.items.copy()
        self.rng.shuffle(shuffled_items)
        for i in range(self.no_of_items):
            item = shuffled_items[i]
            free_caps = np.array(free_bin_caps)
            new_caps = (free_caps - item)
            valid_idx = np.where(new_caps >= 0)[0]
            if valid_idx.size:
                idx = valid_idx[new_caps[valid_idx].argmin()]
                bins[idx].append(item)
                free_bin_caps[idx] -= item
            else:
                bins.append([item])
                free_bin_caps.append(self.max_bin_capacity - item)
        
        new_bin = BinDetails(bins, free_bin_caps)
        if gen:
            return new_bin

        self.bin_solutions[id] = new_bin