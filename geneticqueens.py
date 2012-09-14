from abc import ABCMeta, abstractmethod
from collections import defaultdict
import random, math, sys, itertools

class GeneticQueens:
    __metaclass__ = ABCMeta

    def __init__(self, board_size, population_size, selection_rate, mutation_probability):
        self.board_size = board_size
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_probability = mutation_probability

    @abstractmethod
    def generate_random_individual():
        pass

    @abstractmethod
    def attacking_queen_pairs(self, chromosome):
        pass

    @abstractmethod
    def mate(self, first_parent, second_parent):
        pass

    def simulate(self, max_generations):
        self.population = sorted([(self.attacking_queen_pairs(individual), individual) for individual in (self.generate_random_individual() for x in xrange(self.population_size))])
        
        generations = 1
        while self.population[0][0] != 0 and generations < max_generations: 

            n_to_select = int(math.ceil(self.population_size * self.selection_rate))
            selected = [item[1] for item in self.population[:n_to_select]]

            total_conflicts = 0
            for item in self.population:
                total_conflicts += item[0]

            print "Generation " + str(generations) + " selection average attacking pairs: " + str(total_conflicts / n_to_select)
            print str(len(set([tuple(x[1]) for x in self.population])))

    
            perms = itertools.permutations(selected, 2)
            
            new_population = []
            i = 0
            while i < self.population_size:
                try:
                    perm = perms.next()
                except StopIteration:
                    break
                individual = self.mate(*perm)
                new_population.append((self.attacking_queen_pairs(individual), individual))
                i += 1

            i = 0
            while len(new_population) < self.population_size:
                new_population.append(self.population[i])
                i += 1

                
            self.population = sorted(new_population)

            generations += 1

        return (self.population[0][0], generations)


class Original(GeneticQueens):
    def generate_random_individual(self):
        return [random.randint(0, self.board_size - 1) for x in xrange(self.board_size)]

    def mate(self, first_parent, second_parent):
        crossover_split_index = random.randint(1, self.board_size - 1)

        child = first_parent[:crossover_split_index] + second_parent[crossover_split_index:]

        if random.random() < self.mutation_probability:
            child = self.mutate(child)

        return child

    def attacking_queen_pairs(self, chromosome):
        attacking_pairs = 0;

        horizontal = lambda row, column: row
        topbottom_diagonal = lambda row, column: row - column
        bottomtop_diagonal = lambda row, column: row + column

        for rank in [horizontal, topbottom_diagonal, bottomtop_diagonal]:
            attack_map = defaultdict(lambda: 0)
            for col, row in enumerate(chromosome):
                attacking_pairs += attack_map[rank(row, col)]
                attack_map[rank(row, col)] += 1

        return attacking_pairs

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
        return result

class RooksOnly(Original):
    def generate_random_individual(self):
        return random.sample(range(self.board_size), self.board_size)

class Relative(Original):
    def generate_random_individual(self):
        return [random.randint(0, (self.board_size - 1) - x) for x in xrange(self.board_size)]

    def convert_chromosome(self, relative_chromosome):
        available = range(self.board_size)
        
        original_style_chromosome = []

        for gene in relative_chromosome:
            original_style_chromosome.append(available.pop(gene))

        return original_style_chromosome
            

    def attacking_queen_pairs(self, chromosome):
        return super(Relative, self).attacking_queen_pairs(self.convert_chromosome(chromosome))
    
    def board_as_string(self, chromosome):
        return super(Relative, self).board_as_string(self.convert_chromosome(chromosome))

    def mutate(self, chromosome):
        result = chromosome
        change_index = random.randint(0, self.board_size - 1)
        result[change_index] = random.randint(0, (self.board_size - 1) - change_index)
        return result

class PermutationFixing(RooksOnly):
    def mate(self, first_parent, second_parent):
        child = first_parent

        change_index = random.randint(1, self.board_size - 1)

        indexes = []
        for item in first_parent[change_index:]:
            indexes.append((second_parent.index(item), item)) 

        indexes.sort()

        for pos, item in enumerate(indexes):
            child[change_index + pos] = item[1]

        return child


    def mutate(self, chromosome):
        result = chromosome
        (a, b) = random.sample(chromosome, 2)
        temp = result[a]
        result[a] = result[b]
        result[b] = temp

        return result

class PermutationGroup(PermutationFixing):
    def mate(self, first_parent, second_parent):
        child = []
        for pos in xrange(self.board_size):
            child.append(first_parent[second_parent[pos]])

        return child

def usage_and_exit():
    print "Usage:"
    print "python geneticqueens.py testalgs boardsize population selectionrate mutationrate maxgenerations trials"
    print "python geneticqueens.py single boardsize"

    exit(1)

def test_algorithms():
    for algorithm in [Original, RooksOnly, Relative, PermutationFixing, PermutationGroup]:
        try:
            size = int(sys.argv[2])
            population = int(sys.argv[3])
            selectionrate = float(sys.argv[4])
            mutationrate = float(sys.argv[5])
            maxgenerations = int(sys.argv[6])
            trials = int(sys.argv[7])
        except:
            usage_and_exit()
                    
        print algorithm.__name__
        sim = algorithm(size, population, selectionrate, mutationrate)
        solution_found = 0
        solution_generations = 0
        for x in xrange(trials):
            (attacking, generations) = sim.simulate(maxgenerations)
            print (attacking, generations)
            if attacking == 0:
                solution_found += 1
                solution_generations += generations

        print "solution %: " + str(solution_found * 100.0 / trials)
        if solution_found > 0:
            print "avg generations: " + str(solution_generations / solution_found)
        print

def single_board():
    try:
        board_size = int(sys.argv[2])
    except:
        usage_and_exit()

    sim = Relative(board_size, 2000, .2, .2)
    
    while True:
        (attacking, generations) = sim.simulate(50)
        print (attacking, generations)
        if attacking == 0:
            print sim.board_as_string(sim.population[0][1])
            break

if __name__ == "__main__":
    if sys.argv[1] == "testalgs":
        test_algorithms()
    elif sys.argv[1] == "single":
        single_board()
    else:
        usage_and_exit()
