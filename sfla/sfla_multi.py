import copy
import sys, logging
import time
import numpy as np
import concurrent.futures
from utils.binpackingsolution import BinPackingSolutions
from utils.binpackingsolution import BinDetails

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s: %(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SFLA:
    def __init__(self, frogs, mplx_no, no_of_iteration, no_of_mutation, q):
        self.frogs = frogs
        self.mplx_no = mplx_no
        self.FrogsEach = int(self.frogs/self.mplx_no)
        self.weights = [2*(self.FrogsEach + 1 - j)/(self.FrogsEach * (self.FrogsEach+1)) for j in range(1, self.FrogsEach+1)] 
        self.no_of_iteration = no_of_iteration
        self.no_of_mutation = no_of_mutation
        self.q = q
        self.bins_data = BinPackingSolutions()
        self.seed_seq = np.random.SeedSequence(98765)
        # self.rng = np.random.default_rng(self.seed_seq)

    def __repr__(self):
        return f"SFLA (Frogs = {self.frogs}, Memeplexes = {self.mplx_no})"
    
    def __str__(self):
        return f"SFLA (Frogs = {self.frogs}, Memeplexes = {self.mplx_no})"

    @property
    def memeplexes(self) -> np.ndarray:
        return self._memeplexes

    @memeplexes.setter
    def memeplexes(self, memeplexes: np.ndarray):
        self._memeplexes = memeplexes

    def find_score(self, id=-1, bin_sol: BinDetails=None, initial=False):
        """Find score using the formula:
            score = 1 - (sum((sum_of_weight[i]/c)^2 for each bin i)/no_of_bins)
        """
        k = 2
        if initial and id != -1:
            no_of_bins = self.bins_data.bin_solutions[id].no_of_bins
            bin_sum = self.bins_data.max_bin_capacity - np.array(self.bins_data.bin_solutions[id].free_bin_caps)
            score = 1 - (np.sum((bin_sum/self.bins_data.max_bin_capacity) ** k)/no_of_bins)
            self.bins_data.bin_solutions[id].score = score
        else:
            no_of_bins = bin_sol.no_of_bins
            bin_sum = self.bins_data.max_bin_capacity - np.array(bin_sol.free_bin_caps)
            score = 1 - (np.sum((bin_sum/self.bins_data.max_bin_capacity) ** k)/no_of_bins)
            return score

    def find_score_multi(self, bin_sol: BinDetails=None):
        """Find score using the formula:
            score = 1 - (sum((sum_of_weight[i]/c)^2 for each bin i)/no_of_bins)
        """
        k = 2
        no_of_bins = bin_sol.no_of_bins
        bin_sum = self.bins_data.max_bin_capacity - np.array(bin_sol.free_bin_caps)
        score = 1 - (np.sum((bin_sum/self.bins_data.max_bin_capacity) ** k)/no_of_bins)
        bin_sol.score = score

    def generate_one_frog(self, frog_args):
        frog_id, seed = frog_args
        bin_sol = self.bins_data.best_fit_heuristic_multi(seed) 
        self.find_score_multi(bin_sol)
        return (frog_id, bin_sol)

    def generate_init_population(self):
        """Generation of initial population
        """
        logger.info(f"Generating initial population (Number of frogs: {self.frogs})")
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            new_seeds = self.seed_seq.spawn(self.frogs)
            args = [[frog_id, seed] for frog_id, seed in zip(range(self.frogs), new_seeds)]
            results = executor.map(self.generate_one_frog, args)
            for r in results:
                if r:
                    self.bins_data.bin_solutions[r[0]] = r[1]

        # for frog_id in range(self.frogs):
        #     self.generate_one_frog(frog_id)

        # for frog_id in range(self.frogs):
        #     self.bins_data.best_fit_heuristic(frog_id) 
        #     self.find_score(frog_id, initial=True)

    def sort_frog(self):
        logger.info(f"Sorting the frogs and making {self.mplx_no} memeplexes with {self.frogs} frogs each")
        sorted_fitness = np.array(sorted(
            self.bins_data.bin_solutions, key=lambda x: self.bins_data.bin_solutions[x].score))
        
        memeplexes = np.empty((self.mplx_no, self.FrogsEach))
        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                memeplexes[i, j] = sorted_fitness[i + (self.mplx_no*j)]
        return memeplexes

    def find_old_bin_id(self, worst_frog, item):
        new_locations = [id for id, bins in enumerate(worst_frog) if item in bins]
        return new_locations[0]

    def generate_swap_set(self, best_sol, worst_sol, fw, seed):
        """Calculates next step
        Args:
            best_sol, worst_sol, fw
        Returns:
            swap_set
        """
        rng = np.random.default_rng(seed)
        swap_set = np.array([[i, item] for i in range(len(best_sol)) for item in best_sol[i] if item not in worst_sol[i]])
        old_size = swap_set.shape[0]
        new_size = int(fw * old_size)
        idxs = rng.permutation(old_size)[:new_size]
        swap_set = swap_set[idxs]
        return swap_set

    def new_step(self, best_frog: BinDetails, worst_frog: BinDetails, seed):
        """Calculates next step
        Args:
            best_frog: best frog 
            worst_frog: worst frog
        
        Returns:
            new_frog: mutated Bin Solution
        """
        new_seed = seed.spawn(1)
        fw = best_frog.score/worst_frog.score
        swap_set = self.generate_swap_set(best_frog.bins, worst_frog.bins, fw, new_seed[0])

        new_sol = copy.deepcopy(worst_frog.bins)
        new_free_bin = copy.deepcopy(worst_frog.free_bin_caps)
        
        for bin_id, item in swap_set:
            if new_free_bin[bin_id] >= item:
                old_bin_id = self.find_old_bin_id(new_sol, item)
                new_sol[bin_id].append(item)
                new_free_bin[bin_id] -= item
                new_sol[old_bin_id].remove(item)
                new_free_bin[old_bin_id] += item

        idxs = [i for i, size in enumerate(new_free_bin) if size == self.bins_data.max_bin_capacity]
        new_free_bin = [size for i, size in enumerate(new_free_bin) if i not in idxs]
        new_sol = [bin_items for i, bin_items in enumerate(new_sol) if i not in idxs]

        new_frog = BinDetails(bins=new_sol, free_bin_caps=new_free_bin)
        return new_frog

    def local_search_one_memeplex(self, ls_args):
        """
        Args:
            im: current memeplex index
            iter_idx: current iteration index
        
        Returns:
            extracted_bin_sols: modified bin solutions
            im: current memeplex index
            memeplex: modified memeplex
        """
        im, iter_idx, seed = ls_args
        rng = np.random.default_rng(seed)
        memeplex = self.memeplexes[im]
        extracted_bin_sols = {int(item):self.bins_data.bin_solutions.get(item) for item in memeplex}

        for idx in range(self.no_of_mutation):
            logger.info(f"Iteration {iter_idx} -- Local Search of Memeplex {im + 1}: Mutation {idx + 1}/{self.no_of_mutation}")
            rValue = rng.random(self.FrogsEach) * self.weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])
            submemeplex = memeplex[subindex] 

            Pb = extracted_bin_sols[int(submemeplex[0])]
            Pw = extracted_bin_sols[int(submemeplex[self.q - 1])]
            
            globStep = False
            censorship = False
            
            logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}: Learn from local best Pb")
            new_frog = self.new_step(Pb, Pw, seed)
            self.find_score_multi(new_frog)
            if new_frog.score > Pw.score:
                globStep = True     
            
            if globStep:
                logger.info(
                    f"Iteration {iter_idx} -- Memeplex {im + 1}: Score didn't improve... Learn from global best Pb")
                new_frog = self.new_step(self.frog_gb, Pw, seed)
                self.find_score_multi(new_frog)
                if new_frog.score > Pw.score:
                    censorship = True

            if censorship:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}: Still score didn't improve... generating a new frog")
                new_frog = self.bins_data.best_fit_heuristic_multi(seed)
                self.find_score_multi(new_frog)

            extracted_bin_sols[int(submemeplex[self.q-1])] = new_frog
            memeplex = np.array(sorted(extracted_bin_sols, key = lambda x: extracted_bin_sols[x].score))
            logger.info(f"Iteration {iter_idx} -- Local Search of Memeplex {im + 1}: Bin Solution moved to Bin_ID -> {int(submemeplex[self.q-1])} ::: {new_frog} ::: Mutation {idx + 1}/{self.no_of_mutation} finished!!")
        
        return (extracted_bin_sols, im, memeplex)
        
    def local_search(self, iter_idx):
        """Local Search
        Args:
            iter_idx: current iteration index
        """
        self.frog_gb = self.bins_data.bin_solutions.get(int(self.memeplexes[0][0]))
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            new_seeds = self.seed_seq.spawn(self.mplx_no)
            ls_args = [[frog_id, iter_idx, seed] for frog_id, seed in zip(range(self.mplx_no), new_seeds)]
            results = executor.map(self.local_search_one_memeplex, ls_args)
            
            for r in results:
                if r:
                    self.bins_data.bin_solutions.update(r[0])
                    self.memeplexes[r[1]] = r[2]


    def shuffle_memeplexes(self):
        """Shuffles the memeplexes and sorting them.
        """
        logger.info("Shuffling the memeplexes and sorting them")
        temp = self.memeplexes.flatten()
        temp = np.array(sorted(temp, key = lambda x: self.bins_data.bin_solutions.get(x).score))
        for j in range(self.FrogsEach):
            for i in range(self.mplx_no):
                self.memeplexes[i, j] = temp[i + (self.mplx_no * j)]
                
    def run_sfla(self, data_path):
        logger.info("Starting SFLA algorithm")
        self.data_path = data_path
        self.bins_data.extract_from_file(self.data_path)
        s1 = time.time()
        self.generate_init_population()
        self.memeplexes = self.sort_frog()
        for idx in range(self.no_of_iteration):
            logger.info(f"Local Search: {idx+1}/{self.no_of_iteration}")
            self.local_search(idx+1)
            self.shuffle_memeplexes()
        e1 = time.time()
        logger.info(f"Time taken: {e1-s1}")
        logger.info(f"Memeplexes :::\n{self.memeplexes} ::: Best Frog => {self.bins_data.bin_solutions.get(self.memeplexes[0][0])}")
        logger.info(f"Best Frog Bins => {self.bins_data.bin_solutions.get(self.memeplexes[0][0]).bins}")
        logger.info(f"Best Frog free capacities in bins => {self.bins_data.bin_solutions.get(self.memeplexes[0][0]).free_bin_caps}")


if __name__ == "__main__":
    n = 100
    path = "./../data/bin1data/N1C1W1_A.BPP"
    sfla = SFLA(frogs=400, mplx_no=40, no_of_iteration=n, no_of_mutation=12, q=8)  
    sfla.run_sfla(path)