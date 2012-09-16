from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
import random, math, sys, itertools

class GeneticAlgorithm:
    __metaclass__ = ABCMeta

    IndividualWithScore = namedtuple('IndividualWithScore', ['score', 'individual'])

    def __init__(self, problem_representation, population_size, 
            selection_rate, mutation_probability):

        self.problem_representation = problem_representation
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_probability = mutation_probability

    def annotate_population(self, population):
        return sorted(
                [GeneticAlgorithm.IndividualWithScore(
                    self.problem_representation.score(individual), tuple(individual)) 
                    for individual in population])

    def solution_found(self):
        return self.population[0].score == 0

    @abstractmethod
    def reproduce(self, population):
        pass 

    def mutate_some(self, population):
        for individual in population:
            if random.random() < self.mutation_probability:
                yield self.problem_representation.mutate(individual)
            else:
                yield individual

    def simulate(self, max_generations):
        self.population = self.annotate_population(
                (tuple(self.problem_representation.generate_random_individual()) for x in xrange(self.population_size)))
        
        generation = 1
        while not self.solution_found() and generation < max_generations: 

            total_conflicts = 0
            for item in self.population:
                total_conflicts += item[0]

            #print "Generation " + str(generation)
            #print "Average attacking pairs: " + str(total_conflicts)
            #print "Unique invidiuals: " + str(len(set(self.population)))
            #print

            self.population = self.annotate_population(self.mutate_some(self.reproduce(self.population)))

            generation += 1

        return (self.population[0][0], generation)

class RandomPermutations(GeneticAlgorithm):
    def select_best_parents(self, population):
        n_to_select = int(math.ceil(self.population_size * self.selection_rate))

        return [item[1] for item in self.population[:n_to_select]]

    def reproduce(self, population):
        selected = self.select_best_parents(population)

        return (self.problem_representation.crossover(*random.sample(selected, 2))
                    for x in xrange(self.population_size))

class Tournament(GeneticAlgorithm):
    def select_individual(self, population):
        return sorted(random.sample(population, min(10, len(population))))[0][1]

    def reproduce(self, population):
        new_population = set()

        while len(new_population) < self.population_size:
            new_population.add(self.problem_representation.crossover(self.select_individual(population), self.select_individual(population)))

        return new_population


class PermutationsPlusOldGeneration(RandomPermutations):
    def reproduce(self, population):
        selected = self.select_best_parents(population)

        all_permutations = [permutation for permutation in itertools.permutations(selected, 2)]

        all_possible_children = set([self.problem_representation.crossover(*permutation)
                for permutation in all_permutations])
        
        new_population = set(random.sample(all_possible_children,
                min(len(all_possible_children), self.population_size)))

        while len(new_population) < self.population_size:
            new_population.add(population.pop(0).individual)

        return new_population


class ProblemRepresentation:
    __metaclass__ = ABCMeta

    def __init__(self, board_size):
        self.board_size = board_size

    @abstractmethod
    def generate_random_individual():
        pass

    @abstractmethod
    def score(self, chromosome):
        pass

    @abstractmethod
    def crossover(self, first_parent, second_parent):
        pass


class NQueens(ProblemRepresentation):
    def generate_random_individual(self):
        return [random.randint(0, self.board_size - 1) for x in xrange(self.board_size)]

    def crossover(self, first_parent, second_parent):
        crossover_split_index = random.randint(1, self.board_size - 1)

        child = first_parent[:crossover_split_index] + second_parent[crossover_split_index:]

        return tuple(child)

    def score(self, chromosome):
        score = 0;

        horizontal = lambda row, column: row
        topbottom_diagonal = lambda row, column: row - column
        bottomtop_diagonal = lambda row, column: row + column

        for rank in [horizontal, topbottom_diagonal, bottomtop_diagonal]:
            attack_map = defaultdict(lambda: 0)
            for col, row in enumerate(chromosome):
                score += attack_map[rank(row, col)]
                attack_map[rank(row, col)] += 1

        return score

    def board_as_string(self, positions):
        result = ""
        for row in range(len(positions)):
            for column in range(len(positions)):
                if positions[column] == row:
                    result = result + "Q"
                else:
                    result = result + "_"
                result += " "
            result += "\n"
        return result
            

    def mutate(self, chromosome):
        result = list(chromosome)
        change_index = random.randint(0, self.board_size - 1)
        result[change_index] = random.randint(0, self.board_size - 1)
        return tuple(result)


class Relative(NQueens):
    def generate_random_individual(self):
        return [random.randint(0, (self.board_size - 1) - x) for x in xrange(self.board_size)]

    def convert_chromosome(self, relative_chromosome):
        available = range(self.board_size)
        
        original_style_chromosome = []

        for gene in relative_chromosome:
            original_style_chromosome.append(available.pop(gene))

        return tuple(original_style_chromosome)
            

    def score(self, chromosome):
        return super(Relative, self).score(self.convert_chromosome(chromosome))
    
    def board_as_string(self, chromosome):
        return super(Relative, self).board_as_string(self.convert_chromosome(chromosome))

    def mutate(self, chromosome):
        result = list(chromosome)
        change_index = random.randint(0, self.board_size - 1)
        result[change_index] = random.randint(0, (self.board_size - 1) - change_index)
        return tuple(result)


class RooksOnly(NQueens):
    def generate_random_individual(self):
        return random.sample(range(self.board_size), self.board_size)


class PermutationFixing(RooksOnly):
    def crossover(self, first_parent, second_parent):
        child = list(first_parent)

        change_index = random.randint(1, self.board_size - 1)

        indexes = []
        for item in first_parent[change_index:]:
            indexes.append((second_parent.index(item), item)) 

        indexes.sort()

        for pos, item in enumerate(indexes):
            child[change_index + pos] = item[1]

        return tuple(child)


    def mutate(self, chromosome):
        result = list(chromosome)
        (a, b) = random.sample(chromosome, 2)
        temp = result[a]
        result[a] = result[b]
        result[b] = temp

        return tuple(result)


class PermutationGroup(PermutationFixing):
    def crossover(self, first_parent, second_parent):
        child = []
        for pos in xrange(self.board_size):
            child.append(first_parent[second_parent[pos]])

        return tuple(child)



def usage_and_exit():
    print 'Usage:'
    print 'python geneticqueens.py testreps boardsize population selectionrate mutationrate maxgenerations trials'
    print 'python geneticqueens.py testalgs boardsize population selectionrate mutationrate maxgenerations trials'
    print 'python geneticqueens.py single boardsize'

    exit(1)

def test_sim(sim, sim_name, maxgenerations, trials):
    print sim_name

    solutions_found = 0
    total_solution_generations = 0

    for trial in xrange(trials):
        (attacking_pairs, generations) = sim.simulate(maxgenerations)
        print (attacking_pairs, generations)
        if attacking_pairs == 0:
            solutions_found += 1
            total_solution_generations += generations

    print '%d percent of trials produced a solution.' % int(solutions_found * 100.0 / trials)

    if solutions_found > 0:
        print 'Average %d generations to produce a solution.' % (total_solution_generations / solutions_found)
    print
    print


def test_algorithms():
    try:
        board_size = int(sys.argv[2])
        population = int(sys.argv[3])
        selectionrate = float(sys.argv[4])
        mutationrate = float(sys.argv[5])
        maxgenerations = int(sys.argv[6])
        trials = int(sys.argv[7])
    except:
        usage_and_exit()

    for algorithm in [Tournament, RandomPermutations, PermutationsPlusOldGeneration]:
        rep = Relative(board_size)
        sim = algorithm(rep, population, selectionrate, mutationrate)
        test_sim(sim, algorithm.__name__, maxgenerations, trials)

def test_problem_representations():
    try:
        board_size = int(sys.argv[2])
        population = int(sys.argv[3])
        selectionrate = float(sys.argv[4])
        mutationrate = float(sys.argv[5])
        maxgenerations = int(sys.argv[6])
        trials = int(sys.argv[7])
    except:
        usage_and_exit()

    for problem_representation in [NQueens, RooksOnly, Relative, 
            PermutationFixing, PermutationGroup]:
                    
        rep = problem_representation(board_size)
        sim = Tournament(rep, population, selectionrate, mutationrate)
        test_sim(sim, problem_representation.__name__, maxgenerations, trials)

def test_problem_representations_large():
    try:
        board_size = int(sys.argv[2])
        population = int(sys.argv[3])
        selectionrate = float(sys.argv[4])
        mutationrate = float(sys.argv[5])
        maxgenerations = int(sys.argv[6])
        trials = int(sys.argv[7])
    except:
        usage_and_exit()

    for problem_representation in [Relative, 
            PermutationFixing]:
                    
        rep = problem_representation(board_size)
        sim = Tournament(rep, population, selectionrate, mutationrate)
        test_sim(sim, problem_representation.__name__, maxgenerations, trials)

def single_board():
    try:
        board_size = int(sys.argv[2])
    except:
        usage_and_exit()

    rep = PermutationFixing(board_size)
    sim = Tournament(rep, 2000, .2, .4)
    
    while True:
        (attacking, generations) = sim.simulate(50)
        print (attacking, generations)
        if attacking == 0:
            print sim.problem_representation.board_as_string(sim.population[0][1])
            break

if __name__ == "__main__":
    try: 
        operation = sys.argv[1]
    except:
        usage_and_exit()

    if operation == 'testreps':
        test_problem_representations()
    elif operation == 'testrepslg':
        test_problem_representations_large()
    elif operation == 'testalgs':
        test_algorithms()
    elif operation == 'single':
        single_board()
    else:
        usage_and_exit()
