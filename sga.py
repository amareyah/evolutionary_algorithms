import numpy as np
from typing import Callable
from operator import itemgetter


class SGA(object):
    def __init__(
        self,
        obj_func: Callable,
        population_size: int,
        chromo_size: int,
        x_min: float,
        x_max: float,
    ):
        self.objective_func = obj_func
        self.POPULATION_SIZE = population_size
        self.CHROMOSOME_SIZE = chromo_size
        self.x_MIN = x_min
        self.x_MAX = x_max
        self.initialize_genotype_population()

    def initialize_genotype_population(self):
        """
        Creates initial population.
        """
        self.population = np.random.randint(
            low=0, high=2, size=(self.POPULATION_SIZE, self.CHROMOSOME_SIZE), dtype=bool
        )

    @staticmethod
    def binary_to_float(bin_arr, a, b):
        """Converts binary array to float array.
        bin_arr: (d0, d1, d2) dimensional np.array, where 
        d0 - population size, 
        d1 - parameter dimension, 
        d2 - chromosome number.

        a, b : d1 dimensional vectors, each component represent low and high boundaries for each dimension.

        Returns: (d0, d1) dimensional float type array.
        """
        bit_num = bin_arr.shape[-1]

        dec = bin_arr @ np.power(2, np.arange(bit_num))

        zero_one_range = dec / (2 ** bit_num)

        real_arr = zero_one_range * (b - a) + a

        return real_arr

    def get_fittenss_values(self, genotype: np.ndarray):
        """
        Calculates fittness values for genotype population.
        """
        x_phenotype = self.binary_to_float(genotype, self.x_MIN, self.x_MAX)

        return self.objective_func(x_phenotype)

    @staticmethod
    def get_selection_probs(fittness_vals):
        """Calculates individuals probability for mating selection using their fittness values.
        Returns array of probabilities.
        """
        if len(fittness_vals.shape) > 1:
            raise Exception(
                f"fittness_vals should be one dimensional array, but it is {fittness_vals.shape}"
            )

        return fittness_vals / np.sum(fittness_vals)

    def create_mating_pool(self):

        self.population_fittness = self.get_fittenss_values(self.population)

        probs = self.get_selection_probs(self.population_fittness)

        indeces = np.random.choice(
            np.arange(self.POPULATION_SIZE),
            size=self.POPULATION_SIZE,
            replace=True,
            p=probs,
        )

        self.mating_pool = self.population[indeces].copy()
        self.mating_pool_fittness = self.population_fittness[indeces].copy()
        """ TODO: should we keep separately population and separately mating_pool? 
        we could overwrite mating_pool to population and mating_pool will become population on which we perform variation operations. """

    def shuffle_mating_pool(self):
        """
        Shuffles mating pool and mating pool fitness array equaly.
        """
        shuffled_index = np.random.permutation(self.POPULATION_SIZE)

        self.mating_pool = self.mating_pool[shuffled_index]

        self.mating_pool_fittness = self.mating_pool_fittness[shuffled_index]

    @staticmethod
    def split_to_chunks(x: np.ndarray, n: int = 2):
        """Splits array into evenly sized chunks.
        
        Keyword Arguments:
            n {int} -- number of elements in chunks (default: {2})
        
        Yields:
            ndarray -- copy of n-sized chunk of individuals from x array
        """
        if len(x) % n != 0:
            raise Exception("length of x should be divisible on n with remainder zero.")

        for i in range(0, len(x), n):
            yield x[i : i + n].copy()

    @staticmethod
    def single_point_crossover(x: np.ndarray):
        """Performs single point crossover on two given individuals
        
        Arguments:
            x {np.ndarray} -- genotype array of shape (2, chromosome_size).
        
        Returns:
            {np.ndarray} -- crossovered genotype array of shape (2, chromosome_size).
        """
        x0 = x[0]
        x1 = x[1]

        point = np.random.randint(low=1, high=len(x[1]))

        x1[:point], x0[:point] = x0[:point], x1[:point].copy()

        return x

    @staticmethod
    def flip_bit_mutate(x: np.ndarray, mutation_rate: float = 0.1):
        """Performs mutation of each gene with probability {mutation_rate} of genotype individuals (in_place).
        
        Arguments:
            x {np.ndarray} -- array of genotype individuals
        
        Keyword Arguments:
            mutation_rate {float} -- probability of gene mutation (default: {0.1})
        
        Returns:
            {np.ndarray} -- mutateted individuals
        """
        probs = np.random.rand(x.shape[0], x.shape[1])

        mutation_mask = probs < mutation_rate

        x[mutation_mask] = ~x[mutation_mask]

        return x

    def perform_variation(
        self, crossover_rate: float = 0.5, mutation_rate: float = 0.4
    ):
        """Performs variation operations on mating pool.
        
        Keyword Arguments:
            crossover_rate {float} -- Number bewteen 0 and 1. Probability that given two individuals 
                                      perform a crossover. If two individuals do not perform a crossover 
                                      according to probability crossover_rate, their off-spring are themselves. 
                                      (default: {0.5})
            mutation_rate {float} -- probability of mutation of each gene. (default: {0.4})
        
        Returns:
            {np.ndarray} -- [description]
        """

        offsprings = []

        self.create_mating_pool()

        self.shuffle_mating_pool()

        for chunk in self.split_to_chunks(self.mating_pool, n=2):
            if np.random.rand() < crossover_rate:
                chunk = self.single_point_crossover(chunk)

            chunk = self.flip_bit_mutate(chunk, mutation_rate=mutation_rate)

            offsprings.extend(chunk)

        return np.array(offsprings)

    def perform_selection(self, offsprings: np.ndarray):
        """Performs selection of new generation.
        We unite old generation with offsprings, sort them by fitness values
        and then take top POPULATION_NUMBER of individuals as next generation.
        It sets instance fields {population} and {population_fittness} to new values.
        
        Arguments:
            offsprings {np.ndarray} -- genotype array of offsprings.
        """
        offsprings = self.perform_variation(crossover_rate=0.5, mutation_rate=0.4)
        offsprings_fittness = self.get_fittenss_values(offsprings)

        z1 = list(zip(offsprings, offsprings_fittness))
        z2 = list(zip(self.mating_pool, self.mating_pool_fittness))

        z3 = [*z1, *z2]

        z3.sort(key=itemgetter(1), reverse=True)

        z3 = z3[: self.POPULATION_SIZE]

        self.population = np.array([inds for inds, _ in z3])
        self.population_fittness = np.array([fittness for _, fittness in z3])

    @staticmethod
    def get_best_solution(population: np.ndarray, fittness_vals: np.ndarray):
        """Returns x and corresponding max fitness value from given population and fittness value arrays. 
        
        Arguments:
            population {np.ndarray} -- population of individuals
            fittness_vals {np.ndarray} -- [individuals fitness values]
        
        Returns:
            {Tuple}: best_individual, best_fit_value
        """
        best_fit_value = np.max(fittness_vals)
        best_fit_value_index = np.argmax(fittness_vals)
        best_individual = population[best_fit_value_index]

        return best_individual, best_fit_value

    def optimize(
        self,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        max_iter: int = 1000,
        precision: float = 0.0001,
        variance_elem_number: int = 20,
    ):
        """Performs optimization process of given objective function using simple genetic algorithm.
        
        Keyword Arguments:
            crossover_rate {float} -- crossover rate. values from [0, 1]. (default: {0.7})
            mutation_rate {float} -- mutation rate. values from [0, 1].  (default: {0.1})
            max_iter {int} -- max number of iterations, after which process stops. (default: {1000})
            precision {float} --  criteria to stop iteration process. 
            When np.abs(variance) of last [variance_elem_number] fitness values is less than precision value, 
            iteration process stops. (default: {0.0001})
            variance_elem_number {int} -- number of fitness elements to calculate variance (default: {20})
        
        Returns:
            {Tuple} -- tuple of arrays of best_solutions and fitness values over the iterations.
        """
        x_solutions = []
        y_obj_func_vals = []

        for i in range(max_iter):

            offsprings = self.perform_variation(crossover_rate, mutation_rate)
            self.perform_selection(offsprings)

            x, y = self.get_best_solution(self.population, self.mating_pool_fittness)
            x_solutions.append(x)
            y_obj_func_vals.append(y)

            if i % 10 == 0:
                print(f"iteration: {i}  objective_func: {y}")

            if (
                precision is not None
                and len(y_obj_func_vals) > variance_elem_number
                and np.abs(np.var(y_obj_func_vals[-variance_elem_number:])) < precision
            ):
                break

        x_solutions = np.array(x_solutions)
        y_obj_func_vals = np.array(y_obj_func_vals)

        return (
            self.binary_to_float(x_solutions, a=self.x_MIN, b=self.x_MAX),
            y_obj_func_vals,
        )
