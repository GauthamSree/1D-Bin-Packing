import sys, logging
import copy, time
import numpy as np

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
        self.rng = np.random.default_rng(98765)

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

    def find_score(self, bin_sol: BinDetails=None):
        """Find score using the formula:
            score = 1 - (sum((sum_of_weight[i]/c)^2 for each bin i)/no_of_bins)
        """
        k = 2
        no_of_bins = bin_sol.no_of_bins
        bin_sum = self.bins_data.max_bin_capacity - np.array(bin_sol.free_bin_caps)
        score = 1 - (np.sum((bin_sum/self.bins_data.max_bin_capacity) ** k)/no_of_bins)
        bin_sol.score = score

    def generate_one_frog(self):
        bin_sol = self.bins_data.best_fit_heuristic() 
        self.find_score(bin_sol)
        return bin_sol

    def generate_init_population(self):
        """Generation of initial population
        """
        logger.info(f"Generating initial population (Number of frogs: {self.frogs})")
                
        for frog_id in range(self.frogs):
            res_bin = self.generate_one_frog()
            self.bins_data.bin_solutions[frog_id] = res_bin

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
        idx = self.rng.permutation(len(new_locations))[0]
        return new_locations[idx]

    def generate_swap_set(self, best_sol, worst_sol, fw):
        """Calculates next step
        Args:
            best_sol, worst_sol, fw
        Returns:
            swap_set
        """
        swap_set = np.array([[i, item] for i in range(len(best_sol)) for item in best_sol[i] if item not in worst_sol[i]])
        old_size = swap_set.shape[0]
        new_size = int(fw * old_size)
        idxs = self.rng.permutation(old_size)[:new_size]
        swap_set = swap_set[idxs]
        return swap_set

    def generate_new_bin_solution(self, worst_frog: BinDetails, swap_set: list):
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
    
    def new_step(self, best_frog: BinDetails, worst_frog: BinDetails):
        """Calculates next step
        Args:
            best_frog: best frog 
            worst_frog: worst frog
        
        Returns:
            new_frog: mutated Bin Solution
        """
        fw = best_frog.score/worst_frog.score
        swap_set = self.generate_swap_set(best_frog.bins, worst_frog.bins, fw)
        new_frog = self.generate_new_bin_solution(worst_frog, swap_set)
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
        im, iter_idx = ls_args
        memeplex = self.memeplexes[im]
        extracted_bin_sols = {int(item):self.bins_data.bin_solutions.get(item) for item in memeplex}

        for idx in range(self.no_of_mutation):
            logger.info(f"Iteration {iter_idx} -- Local Search of Memeplex {im + 1}: Mutation {idx + 1}/{self.no_of_mutation}")
            rValue = self.rng.random(self.FrogsEach) * self.weights
            subindex = np.sort(np.argsort(rValue)[::-1][0:self.q])
            submemeplex = memeplex[subindex] 

            Pb = extracted_bin_sols[int(submemeplex[0])]
            Pw = extracted_bin_sols[int(submemeplex[self.q - 1])]
            
            globStep = False
            censorship = False
            
            logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}: Learn from local best Pb")
            new_frog = self.new_step(Pb, Pw)
            self.find_score(new_frog)
            if new_frog.score > Pw.score:
                globStep = True     
            
            if globStep:
                logger.info(
                    f"Iteration {iter_idx} -- Memeplex {im + 1}: Score didn't improve... Learn from global best Pb")
                new_frog = self.new_step(self.frog_gb, Pw)
                self.find_score(new_frog)
                if new_frog.score > Pw.score:
                    censorship = True

            if censorship:
                logger.info(f"Iteration {iter_idx} -- Memeplex {im + 1}: Still score didn't improve... generating a new frog")
                new_frog = self.generate_one_frog()

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
        for frog_id in range(self.mplx_no):
            res = self.local_search_one_memeplex([frog_id, iter_idx])
            self.bins_data.bin_solutions.update(res[0])
            self.memeplexes[res[1]] = res[2]

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
        best_solution = self.bins_data.bin_solutions.get(self.memeplexes[0][0])
        logger.info(f"Time taken: {e1-s1}s")
        logger.info(f"Memeplexes :::\n{self.memeplexes} ::: Best Frog => {best_solution}")
        logger.info(f"Best Frog Bins => {best_solution.bins}")
        logger.info(f"Best Frog free capacities in bins => {best_solution.free_bin_caps}")
        
        with open("Result.txt", 'w') as result:
            result.write("<==== RESULTS ====>\n")
            result.write(f"Best minimum no of bins and Bin Efficiency by SFLA (Best Frog):- {best_solution}\n")
            result.write(f"<--- Best Frog Bin Solution --->\n")
            for bin_id, bin in enumerate(best_solution.bins):
                result.write(f"Bin {bin_id + 1}: {bin}\n")
            result.write(f"Free capacities in each bins: {best_solution.free_bin_caps}\n")


if __name__ == "__main__":
    n = 100
    path = "./../data/bin1data/N3C2W4_T.BPP"
    # path = "./../data/bin2data/N2W1B1R7.BPP"
    # path = "./../data/bin2data/N3W1B3R0.BPP"
    # path = "./../data/bin2data/N1W1B1R5.BPP"
    # path = "./../data/bin3data/HARD9.BPP"
    sfla = SFLA(frogs=480, mplx_no=40, no_of_iteration=n, no_of_mutation=20, q=8)
    sfla.run_sfla(path)