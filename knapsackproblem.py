import random
import pandas as pd
import numpy as np

def initialize_population(num_items, population_size):
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(num_items)]
        population.append(individual)
    return population


def evaluate_fitness(individual, values, weights, max_weight):
    total_value = 0
    total_weight = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            total_value += values[i]
            total_weight += weights[i]
            if total_weight > max_weight:
                return 0  # Individu melebihi batas berat, fitness = 0
    return total_value  # Fitness = total nilai


def tournament_selection(population, values, weights, max_weight, tournament_size):
    tournament = random.sample(population, tournament_size)
    best_individual = None
    best_fitness = -1
    for individual in tournament:
        fitness = evaluate_fitness(individual, values, weights, max_weight)
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual
    return best_individual


def crossover(parent1, parent2):
    crossover_point = len(parent1) - 3
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def evaluate_individual(individual):
    total_weight = 0
    total_value = 0

    for i in range(len(individual)):
        if individual[i] == 1:
            total_weight += weights[i]
            total_value += values[i]

    return total_weight, total_value


def tablee(population, values, weights):
    df = pd.DataFrame(population)
    df.columns = ['Barang A', 'Barang B', 'Barang C', 'Barang D', 'Barang E']
    df.index = np.arange(1, len(df) + 1)

    # Menghitung total berat dan total harga untuk setiap individu
    total_weights = []
    total_values = []
    for i in range(len(population)):
        individual = population[i]
        total_weight = sum(weights[j] for j in range(len(individual)) if individual[j] == 1)
        total_value = sum(values[j] for j in range(len(individual)) if individual[j] == 1)
        total_weights.append(total_weight)
        total_values.append(total_value)

    # Menambahkan kolom total berat dan total harga ke dalam DataFrame
    df['Total Berat'] = total_weights
    df['Total Harga'] = total_values

    df = df.style.set_caption("Populasi :")
    display(df)


def item_df(values, weights):
    data = {'Value': values,
            'Weight': weights}
    df1 = pd.DataFrame(data)
    df1 = df1.set_index(pd.Index(range(1, len(df1) + 1)))

    df1 = df1.style.set_caption("Barang :")
    display(df1)


def genetic_algorithm(values, weights, max_weight, population_size, num_generations, tournament_size, crossover_rate, mutation_rate):
    num_items = len(values)
    population = initialize_population(num_items, population_size)

    tablee(population, values, weights)
    best_individual = None
    best_fitness = -1

    for generation in range(num_generations):
        print("")
        print("===================================")
        print(f"Generation {generation + 1}:")
        print("===================================")
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, values, weights, max_weight, tournament_size)
            parent2 = tournament_selection(population, values, weights, max_weight, tournament_size)

            print("Parent 1:", parent1)
            print("Parent 2:", parent2)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)
                print("")
                print("Crossover Occurred")
                print("Child 1:", child1)
                print("Child 2:", child2)
                print("")
            else:
                new_population.append(parent1)
                new_population.append(parent2)
                print("No Crossover Occurred")
                print("")

        population = new_population

        for i, individual in enumerate(population):
            original_individual = individual.copy()
            individual = mutate(individual, mutation_rate)
            fitness = evaluate_fitness(individual, values, weights, max_weight)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
                if individual != original_individual:
                    print("===================================")
                    print(f"Mutation Occurred in Individual {i + 1}")
                    print("Original Individual:", original_individual)
                    print("Mutated Individual :", individual)

    print("After:")
    tablee(population, values, weights)

    best_individual = max(population, key=lambda ind: evaluate_fitness(ind, values, weights, max_weight))
    best_fitness = evaluate_fitness(best_individual, values, weights, max_weight)

    if best_fitness == 0:
        print("Individu Terbaik Melebihi Batas Berat")
        best_individual = None
        best_fitness = -1

    return best_individual, best_fitness


values = [10, 8, 12, 6, 8]
weights = [7, 4, 9, 3, 5]
max_weight = 20
population_size = 32
num_generations = 5
tournament_size = 5
crossover_rate = 0.8
mutation_rate = 0.1
num_items = len(values)

item_df(values, weights)

best_individual, best_fitness = genetic_algorithm(values, weights, max_weight, population_size, num_generations, tournament_size, crossover_rate, mutation_rate)
if best_individual is not None:
    total_weight, total_value = evaluate_individual(best_individual)
    print("\nSolusi Terbaik:")
    print("Individu:", best_individual)
    print("Fitness:", best_fitness)
    print("Total Berat:", total_weight)
    print("Total Harga:", total_value)
else:
    print("\nTidak ada solusi yang memenuhi batas berat")
