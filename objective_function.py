import typing
import shutil
import random
import math
import csv
import ast
from statistics import mean

import ioh

from implementation import GeneticAlgorithm
from itertools import product


class CellularAutomata:
    """Skeleton CA, you should implement this."""

    def __init__(self, rule_number: int, k: int):
        """Intialize the cellular automaton with a given rule number"""
        self.rule_number = rule_number
        self.k_dimension = k

    def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
        """Evaluate for T timesteps. Return Ct for a given C0."""
<<<<<<< HEAD
        return c0


def make_objective_function(ct, rule, t, similarity_method):
=======
        states = [0, 1, 2]  # k = 2 or k = 3

        # Create list with tuples of all possible neighbourhood states
        nbh_sts = list(product(states[0:self.k_dimension], repeat=3))
        nbh_sts_rv = list(reversed(nbh_sts))
        possible_nbh_sts = math.pow(self.k_dimension, 3)  # Calculate total possible neighbourhood states

        rule_converted = self.convert_rule(self.rule_number)  # Convert rule number

        while len(rule_converted) < possible_nbh_sts:
            rule_converted = '0' + rule_converted  # Lengthen rule number to be same length as neighbourhood states

        # Determine for every cell the corresponding neighbourhood state + next cell value
        ct_previous = c0
        for step in range(t):
            nbhs = []
            ct = []
            for i, cell in enumerate(ct_previous):  # Find values of neighbouring cells
                if i == 0:
                    neighbourhood = (0, ct_previous[i], ct_previous[i + 1])
                elif i == len(c0) - 1:
                    neighbourhood = (ct_previous[i - 1], ct_previous[i], 0)
                else:
                    neighbourhood = (ct_previous[i - 1], ct_previous[i], ct_previous[i + 1])
                nbhs.append(neighbourhood)
            for nbh in nbhs:  # Determine neighbourhood + corresponding next cell value
                for j, nbh_r in enumerate(nbh_sts_rv):
                    if nbh == nbh_r:
                        ct.append(int(rule_converted[j]))  # Append cell value to next generation
            ct_previous = ct  # Current generation becomes previous
        return ct_previous

    # Convert decimal number to binary or ternary number
    def convert_rule(self, num):
        quotient = num / self.k_dimension
        remainder = num % self.k_dimension
        if quotient == 0:
            return ""
        else:
            return self.convert_rule(int(quotient)) + str(int(remainder))


def make_objective_function(ct, rule, t, similarity_method, k):
>>>>>>> cf67314e1973fc9912490d16829171ed5c0d7492
    '''Create a CA objective function.'''
    
    if similarity_method == 1:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """Method that calculates the score based on how many similar instances they have"""
            # Initialize score
            score = 0

            if len(ct) == len(ct_prime): 
                for i in range(len(ct)):
                    if ct[i] == ct_prime[i]:
                        score+=1
            else:
                raise IndexError("ct and ct_prime should have same length")

            return score

    else:
        def similarity(ct: typing.List[int], ct_prime: typing.List[int]) -> float:
            """Gives score based on length of longest local similar sequence."""
            best_score = -1
            current_score = 0
            check_point = 0
            sequence_len = len(ct)

            while check_point<sequence_len: 
                for i in range(check_point, sequence_len):
                    if ct[i] == ct_prime[i]:
                        current_score+=1
                    if ct[i] != ct_prime[i]:
                        break
                check_point = i + 1
                if current_score>best_score:
                    best_score = current_score
                    current_score = 0
            
            return best_score

    
    def objective_function(c0_prime: typing.List[int]) -> float:
        """Skeleton objective function. 
        
        You should implement a method  which computes a similarity measure 
        between c0_prime a suggested by your GA, with the true c0 state 
        for the ct state given in the sup. material.

        Parameters
        ----------
        c0_prime: list[int] | np.ndarray
            A suggested c0 state
        
        Returns
        -------
        float
            The similarity of ct_prime to the true ct state of the CA           
        """
<<<<<<< HEAD
        print(c0_prime)
        ca = CellularAutomata(rule)
        # print(ca)
=======
        ca = CellularAutomata(rule, k)
>>>>>>> cf67314e1973fc9912490d16829171ed5c0d7492
        ct_prime = ca(c0_prime, t)
        # print(ct_prime)
        return similarity(ct, ct_prime)

    return objective_function


def example(nreps=10):
    """An example of wrapping a objective function in ioh and collecting data
    for inputting in the analyzer."""

<<<<<<< HEAD
    ct, rule, t = "[0, 0, 0, 1, 0, 0, 0]", None, 5  # Given by the sup. material
=======
    ct, rule, t, k = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 30, 5, 2  # Given by the sup. material
>>>>>>> cf67314e1973fc9912490d16829171ed5c0d7492

    # Create an objective function
    objective_function = make_objective_function(ct, rule, t, 2, k)
    
    # Wrap objective_function as an ioh problem
    problem = ioh.wrap_problem(
        objective_function,
        name="objective_function_ca_1", # Give an informative name 
        dimension=10, # Should be the size of ct
        problem_type="Integer",
        optimization_type=ioh.OptimizationType.MAX,
        lb=0,
        ub=1,         # 1 for 2d, 2 for 3d
    )
    # Attach a logger to the problem
    logger = ioh.logger.Analyzer()
    problem.attach_logger(logger)

    # run your algoritm on the problem
    for _ in range(nreps):
        algorithm = GeneticAlgorithm(10)
        algorithm(problem)
        problem.reset()

    logger.close()

    shutil.make_archive("ioh_data", "zip", "ioh_data")
    shutil.rmtree("ioh_data")

def read_csv(ca_input_file: str) -> typing.List[typing.Tuple]:
    """Function that reads in the ca_input.csv file and returns a list of tuples with the data."""
    ca_input_list = []

    with open(ca_input_file, mode ='r')as file:

        # csv file is read
        ca_input_csv = csv.reader(file)
 
        # displaying the contents of the CSV file
        for i, row in enumerate(ca_input_csv): 
            if i == 0:
                continue # skip header row
            for j in range(3):
                row[j] = int(row[j]) # convert strings to integers
            row[3] = ast.literal_eval(row[3]) # convert string of list to list of numbers
            row = tuple(row) # convert list to tuple
            ca_input_list.append(row)

    return ca_input_list


def run_experiment(nreps=10):
    """Objective function is wrapped in ioh and data is collected for inputting in the analyzer."""

    ca_input_list = read_csv("ca_input.csv")

    k, rule, t, ct = ca_input_list[0] # Given by the csv input 

    # Create an objective function
    objective_function = make_objective_function(ct, rule, t, 1, k)
    
    # Wrap objective_function as an ioh problem
    problem = ioh.wrap_problem(
        objective_function,
        name="objective_function_ca_1", # Give an informative name 
        dimension=len(ct), # size of ct in ca_input.csv
        problem_type="Integer",
        optimization_type=ioh.OptimizationType.MAX,
        lb=0,
        ub=k-1,         # 1 for 2d, 2 for 3d
    )
    # Attach a logger to the problem
    # logger = ioh.logger.Analyzer() TODO: uncomment when making data file for ioh website
    # problem.attach_logger(logger) TODO: uncomment when making data file for ioh website

    score_list = []
    # run your algoritm on the problem
    for _ in range(nreps):
        algorithm = GeneticAlgorithm(budget=500, k_reperesentations=k)
        algorithm(problem)
        score_list.append(problem.state.current_best.y)
        problem.reset()

    objective = problem.meta_data.n_variables
    average_score = mean(score_list)
    print(f'For k={k}, rule={rule}, t={t} and ct={ct}: \n')
    print(f'Average performance percentage over {nreps} repetitions: {round((average_score/objective), 4)*100}% \n')
    # logger.close() TODO: uncomment when making data file for ioh website

    # shutil.make_archive("ioh_data", "zip", "ioh_data") TODO: uncomment when making data file for ioh website
    # shutil.rmtree("ioh_data") TODO: uncomment when making data file for ioh website


if __name__ == "__main__":
    run_experiment()
