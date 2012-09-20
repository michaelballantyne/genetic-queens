from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
import random, math, itertools, argparse

class GeneticAlgorithm:
    __metaclass__ = ABCMeta

    IndividualWithScore = namedtuple('IndividualWithScore', ['score', 'individual'])

    def __init__(self, problem_representation, population_size, 
            selection_rate, mutation_probability, verbose=False):

        self.problem_representation = problem_representation
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_probability = mutation_probability
        self.verbose = verbose

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

            if self.verbose:
                print "Generation " + str(generation)
                print "Average attacking pairs: " + str(total_conflicts)
                print "Unique invidiuals: " + str(len(set(self.population)))
                print

            self.population = self.annotate_population(self.mutate_some(self.reproduce(self.population)))

            generation += 1

        return (self.population[0][0], generation)

class Truncation(GeneticAlgorithm):
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


class TruncationPlusPromotion(Truncation):
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


class Columns(ProblemRepresentation):
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


class Relative(Columns):
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


class RooksOnly(Columns):
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


def run(args):
    print "Using population %d, mutation rate %f, and selection rate %f\n" % (args.population, args.mutation_rate, args.selection_rate)

    for alg in args.algorithms:
        print "Algorithm %s:\n" % alg.__name__
        for rep in args.representations:
            print "Representation %s:" % rep.__name__

            rep = Relative(args.board_size)
            sim = alg(rep, args.population, args.selection_rate, args.mutation_rate, verbose=args.print_generation_details)

            solutions_found = 0
            total_solution_generations = 0

            for trial in xrange(args.trials):
                (attacking_pairs, generations) = sim.simulate(args.generations)
                print (attacking_pairs, generations)
                if attacking_pairs == 0:
                    solutions_found += 1
                    total_solution_generations += generations
                    if args.print_solution is True:
                        print rep.board_as_string(sim.population[0][1])

            print '%d percent of trials produced a solution.' % int(solutions_found * 100.0 / args.trials)

            if solutions_found > 0:
                print 'Average %d generations to produce a solution.' % (total_solution_generations / solutions_found)
            print
            print

if __name__ == "__main__":
    algorithms = [Truncation, Tournament]
    representations = [Columns, Relative, RooksOnly, PermutationFixing, PermutationGroup]

    parser = argparse.ArgumentParser()

    parser.add_argument('board_size', type=int, help="size of chessboard to find a solution for")

    parser.add_argument('-a', nargs='*', dest='algorithms', type=lambda name: dict([(alg.__name__, alg) for alg in algorithms])[name], choices=algorithms, default=[Tournament], help="list of algorithms to test")

    parser.add_argument('-r', nargs='*', dest='representations', type=lambda name: dict([(rep.__name__, rep) for rep in representations])[name], choices=representations, default=[PermutationFixing], help="list of problem representations to test")

    parser.add_argument('trials', nargs='?', type=int, default=1, help="trials. 0 means go until a solution is found.")
    parser.add_argument('generations', nargs='?', type=int, default=20, help="generations limit")
    parser.add_argument('population', nargs='?', type=int, default=2000, help="initial population size")
    parser.add_argument('mutation_rate', nargs='?', type=float, default=.2, help="mutation rate")
    parser.add_argument('selection_rate', nargs='?', type=float, default=.2, help="selection rate")

    parser.add_argument('--print', dest='print_solution', action='store_true')
    parser.add_argument('--generation-details', dest='print_generation_details', action='store_true')
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print
        exit(0)
