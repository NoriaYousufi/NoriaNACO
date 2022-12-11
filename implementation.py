"""NACO assignment 22/23.

This file contains the skeleton code required to solve the first part of the 
assignment for the NACO course 2022. 

You can test your algorithm by using the function `test_algorithm`. For passing A1,
your GA should be able to pass the checks in this function for dimension=100.

## Installing requirements
You are encouraged to use a virtual environment. Install the required dependencies 
(via the command line) with the following command:
    pip install ioh>=0.3.3
"""

import random
import shutil
from typing import List

import ioh


class GeneticAlgorithm:
    """An implementation of the Genetic Algorithm."""

# <<<<<<< HEAD
#     def __init__(self, budget: int, population_size: int=5, offspring_size: int=50,
#         crossover_operator: str='discrete', k_crossover_splits: int=5,
#         mutation_operator: str='adaptive', mutation_rate: float=0.01, mutation_decrease_rate: float=0.01,
#         selection_operator: str='tournament', tournament_size: int=20,
# =======
    def __init__(self, budget: int, population_size: int=20, offspring_size: int=100,
        crossover_operator: str='k_point', k_crossover_splits: int=5, \
        mutation_operator: str='bitflip', mutation_rate: float=0.01, mutation_decrease_rate: float=0.01, \
        selection_operator: str='tournament', tournament_size: int=20, \
        k_reperesentations: int=2
        ) -> None:
        """Construct a new GA object.

        Parameters
        ----------
        budget: int
            The maximum number objective function evaluations
            the GA is allowed to do when solving a problem.
        population_size: int
            Size of the population, default=5, should be larger than 0.
        offspring_size: int 
            Size of the offspring created by recombination, default=50. Should be larger than 0. 
        crossover_operator: str
            Type of crossover operator that will be used. Can choose between 'k_point' and 'discrete'.
        k_crossover_splits: int
            Number of times crossover is applied to parents to create offspring.
            Default is 5. 
        mutation_operator: str
            Type of mutation operator that will be used. Can choose between 'bitflip' and 'adaptive'. 
        mutation_rate: float
            Start rate of mutation, default=0.01
        mutation_decrease_rate: float
            Rate with which mutation decreases for adaptive mutation, with a default of 1%
        selection_operator: str
            Selection type that will be used for the mating selection. 
            Can choose between roulette wheel selection and tournament selection.
        tournament_size: int
            Amount of individuals in tournament when using tournament selection. 
            Default is 20. Has to be larger than 0. 
        k_representations: int
            Number of representations that algorithm can have. Default is k=2, so
            that means representation is {0,1}. For k=3 representation is {0,1,2} etc.

        Notes
        -----
        *   You can add more parameters to this constructor, which are specific to
            the GA or one of the (to be implemented) operators, such as a mutation rate.
        """

        ## Parameters
        self.budget = budget

        ## Check whether parameters are not out of allowed range.
        if (0<population_size) and (0<offspring_size) \
            and (0<tournament_size) and (0<k_crossover_splits) \
            and (0<k_reperesentations):
            self.population_size = population_size 
            self.offspring_size = offspring_size
            self.tournament_size = tournament_size
            self.k_crossover_splits = k_crossover_splits
            self.k_representations = k_reperesentations
        else: 
            raise ValueError("Given size out of range.")

        if mutation_operator == 'adaptive' or mutation_operator == 'bitflip':
            self.mutation_operator = mutation_operator
        else: 
            raise ValueError("Given mutation operator does not exist.")

        if crossover_operator == 'k_point' or crossover_operator == 'discrete':
            self.crossover_operator = crossover_operator
        else: 
            raise ValueError("Given crossover operator does not exist.")

        if selection_operator == 'roulette' or selection_operator == 'tournament':
            self.selection_operator = selection_operator
        else: 
            raise ValueError("Given selection operator does not exist.")

        if (0<=mutation_rate<=1) and (0<=mutation_decrease_rate<=1):
            self.mutation_rate = mutation_rate 
            self.mutation_decrease_rate = mutation_decrease_rate
        else:
            raise ValueError("Rate out of range [0,1]")

    def __call__(self, problem: ioh.problem.Integer) -> ioh.IntegerSolution:
        """Run the GA on a given problem instance.

        Parameters
        ----------
        problem: ioh.problem.Integer
            An integer problem, from the ioh package. This version of the GA
            should only work on binary/discrete search spaces.

        Notes
        -----
        *   This is the main body of you GA. You should implement all the logic
            for this search algorithm in this method. This does not mean that all
            the code needs to be in this method as one big block of code, you can
            use different methods you implement yourself.

        *   For now there is a random search process implemented here, which
            is a placeholder, and just to show you how to call the problem
            class.
        """
        
        ## Initialize structures and variables
        population = [] # A list with the current population
        offspring_population = [] # List with offspring population created from population by recombination
        max_score = problem.meta_data.n_variables # The maximum score that can be reached for the problem.

        ## Initialize a random population.
        population = [random.choices(range(self.k_representations), k=problem.meta_data.n_variables) \
            for _ in range(self.population_size)]
        
        ## Evaluate the first population by creating a sorted list from best to worst solution so far. 
        population = sorted(population, key=lambda i: problem(i), reverse=True)
        
        ## Main for loop of GA ##
        while problem.state.evaluations < self.budget:  # Repeat loop as long as budget allows it 

            ## Crossover operator, choice between k_point_crossover and discrete crossover.
            if self.crossover_operator == 'k_point':
                offspring_population = self.k_point_crossover(population, problem)
            elif self.crossover_operator == 'discrete':
                offspring_population = self.discrete_crossover(population, problem)
            
            ## Mutation operator, choice between bitflip and adaptive mutation. 
            if self.mutation_operator == 'bitflip':
                self.bitflip_mutation(offspring_population, problem)
            if self.mutation_operator == 'adaptive':
                self.adaptive_mutation(offspring_population, problem)
            
            ## Evaluate offspring population by sorting from highest to lowest problem score. 
            ## So evaluated_offspring[0] has current highest score. 
            evaluated_offspring = sorted(offspring_population, key=lambda i: problem(i), reverse=True)

            ## If indivdual with max_score is in the offspring, then return it. 
            if problem(evaluated_offspring[0]) == max_score: 
                print(f'problem solved in {problem.state.evaluations} evaluations')
                return problem.state.current_best

            ## Select population, either with roulette wheel or tournament selection 
            if self.selection_operator == 'roulette':
                population = self.roulette_wheel_selection(evaluated_offspring, problem)
            elif self.selection_operator == 'tournament':
                population = self.tournament_selection(evaluated_offspring)
    
        ## Main loop of GA ##
        return problem.state.current_best

    def roulette_wheel_selection(self, evaluated_population: List[List[int]], problem: ioh.problem.Integer) \
        -> List[List[int]]:

        """Roulette wheel selection is a selection operator that chooses the individuals
        for the next population based on the part they take in on a so called 'roulette wheel'
        that is constructed from the fitness of the individuals."""

        selected_population = [] # Initialize population that will be selected
        roulette_wheel = [] # Initialize roulette wheel
        roulette_wheel_part = 0 # Roulette wheel part is a number between 0 and 1. 
        problem_total_sum = sum([problem(i) for i in evaluated_population]) # Total sum of fitnesses is calculated
        
        ## Constructing the roulette wheel. The individuals in evaluated_population are sorted from 
        ## best to worst fitness. The roulette wheel parts are assigned so that each individual gets 
        ## a part proportional to their fitness compared to the total fitness in the whole population.  
        for i in evaluated_population: 
            individual_score = problem(i)
            roulette_wheel_part += round((individual_score/problem_total_sum), 3) 
            roulette_wheel.append(roulette_wheel_part) 

        ## Selecting individuals from the roulette wheel. A random number between 0 and 1 is generated.
        ## The individuals with larger roulette wheel parts have a higher chance that this random number
        ## is in their part of the roulette wheel. 
        for _ in range(self.population_size):
            random_slice = random.random()
            for i in range(self.offspring_size):
                if roulette_wheel[i]>=random_slice:
                    selected_population.append(evaluated_population[i])
                    break
        return selected_population # Return a selected population of size self.population_size

    def tournament_selection(self, evaluated_population: List[List[int]]) \
        -> List[List[int]]:

        """Tournament selection is a selection operator that chooses the individuals for the next population
        from the offspring_population by choosing a number of participants and picking the fittest of those
        participants for the next generation. The number of participants is defined in self.tournament_size."""

        selected_population = [] # Initialize selected population

        ## Tournament loop. The evaluated population is sorted from best to worst individuals. 
        ## Random integers in range of the offspring size are chosen. The minimum value of those 
        ## integers gives the index in evaluated_population of the best individual in the tournament. 
        for _ in range(self.population_size):
            tournament_participants = random.choices(range(self.offspring_size), k=self.tournament_size)
            best_participant = min(tournament_participants)
            selected_population.append(evaluated_population[best_participant])

        return selected_population

    def k_point_crossover(self, selected_population: List[List[int]], problem: ioh.problem.Integer) \
        -> List[List[int]]:

        """k_point_crossover is a recombination operator that performs crossover k times on two random
        parents from the population. It generates two children per two parents. The number of splits
        is defined in self.k_crossover_splits."""

        offspring_population = [] # Initialize offspring population

        ## Main crossover for loop. Parents are randomly selected and crossed over k times.
        for _ in range(int(self.offspring_size/2)): # 2 children are added to offspring in each loop
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            for _ in range(self.k_crossover_splits):
                split_point = random.randint(1,problem.meta_data.n_variables-1)
                child1 = parent1[:split_point] + parent2[split_point:] 
                child2 = parent2[:split_point] + parent1[split_point:]
                parent1 = child1 
                parent2 = child2
            offspring_population += [child1, child2]
        
        ## If the offspring size is uneven then an extra child needs to be added.
        if self.offspring_size % 2 == 1:
            offspring_population += random.choice(([child1],[child2]))
        
        return offspring_population

    def discrete_crossover(self, selected_population: List[List[int]], problem: ioh.problem.Integer) \
        -> List[List[int]]:

        """Discrete crossover is a recombination operator where only 1 child is generated from 2 parents.
        For each bit in the child the bit of one of the parents is randomly chosen."""

        offspring_population = [] # Initialize offspring_population

        ## Main crossover loop. 
        for _ in range(self.offspring_size):
            child = []
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)

            ## Each bit in the child is either from parent1 or parent2. 
            for i in range(problem.meta_data.n_variables):
                random_parent = random.choice([parent1, parent2])
                child.append(random_parent[i])

            offspring_population += [child]

        return offspring_population


    def bitflip_mutation(self, offspring_population: List[List[int]], problem: ioh.problem.Integer) \
        -> None:

        """Bitflip mutation is a simple mutation operator where each bit in an individual has a chance of
        self.mutation_rate to be flipped from 0 to 1 or from 1 to 0.""" #TODO: beschrijving aanpassen

        for i in range(self.offspring_size):
            for j in range(problem.meta_data.n_variables): 
                if random.random()<self.mutation_rate: # chance of mutation depends on mutation rate
                    new_int = random.choice(range(self.k_representations))
                    #offspring_population[i][j] = int(not offspring_population[i][j]) # bitflip TODO: weghalen
                    offspring_population[i][j] = new_int

    def adaptive_mutation(self, offspring_population, problem):

        """Adaptive mutation is a mutation operator where the individuals with a higher fitness have a smaller
        mutation rate than individuals with a lower fitness. The mutation rate decreases for individuals 
        with a better fitness. The decrease rate is defined in self.mutation_decrease_rate."""

        adaptive_mutation_rate = self.mutation_rate # Initialize adaptive mutation rate
        offspring_population = sorted(offspring_population, key=lambda i: problem(i)) # Sorted from worst to best

        ## Main mutation for loop. The mutation rate slowly decreases as the individuals get better.
        for i in range(self.offspring_size):
            adaptive_mutation_rate -= adaptive_mutation_rate*self.mutation_decrease_rate
            for j in range(problem.meta_data.n_variables): 
                if random.random()<adaptive_mutation_rate: 
                    new_int = random.choice(range(self.k_representations))
                    #offspring_population[i][j] = int(not offspring_population[i][j])
                    offspring_population[i][j] = new_int
                    

def test_algorithm(dimension, instance=1):
    """A function to test if your implementation solves a OneMax problem.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    instance: int
        The instance of the problem. Trying different instances of the problem,
        can be interesting if you want to check the robustness, of your GA.
    """

    budget = int(dimension * 5e3)
    problem = ioh.get_problem("LeadingOnes", instance, dimension, "Integer")
    ga = GeneticAlgorithm(budget)
    solution = ga(problem)

    print("GA found solution:\n", solution)

    assert problem.state.optimum_found, "The optimum has not been reached."
    assert (problem.state.evaluations <= budget), (
        "The GA has spent more than the allowed number of evaluations to "
        "reach the optimum."
    )

    print(f"OneMax was successfully solved in {dimension}D.\n")


def collect_data(dimension=100, nreps=5):
    """OneMax + LeadingOnes functions 10 instances.

    This function should be used to generate data, for A1.

    Parameters
    ----------
    dimension: int
        The dimension of the problem, i.e. the number of search space variables.

    nreps: int
        The number of repetitions for each problem instance.
    """

    budget = int(dimension * 5e2)
    suite = ioh.suite.PBO([1, 2], list(range(1, 11)), [dimension])
    logger = ioh.logger.Analyzer(algorithm_name="GeneticAlgorithm1")
    suite.attach_logger(logger)

    for problem in suite:
        print("Solving: ", problem)

        for _ in range(nreps):
            ga = GeneticAlgorithm(budget)
            ga(problem)
            problem.reset()
    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")


if __name__ == "__main__":
    # Simple test for development purpose
    test_algorithm(10)

    # Test required for A1, your GA should be able to pass this!
    for i in range(3):
        test_algorithm(100)

    # If your implementation passes test_algorithm(100)
    #collect_data(100)
