
"""
ACTIVITIES SLOT NUMBER
Meditation	5
Yoga		6
Workout		1
Study		7
Upbeat		2
dance		3
sleep		4
"""
# Importing the libraries
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Global Variables
num_pop = 10
num_activity = 7
num_bits = 8
num_attributes = 8
population_in_binary = [] * num_pop
population_in_decimal = []
fitness_score_pop = [0]*num_pop
Y = []

############################################### Importing the dataset ###########################################################
dataset = pd.read_csv('Songs.csv')
song_att = dataset.iloc[:, [5, 6, 7, 8, 9, 10, 12, 13]].values
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
song_att[:,:]=sc.fit_transform(song_att[:,:]) 
#song_att = song_att.to_numpy()


############################################ Decimal To Binary Conversion ######################################################

def decimal_to_binary(chromosome):
    sol_bin = ""
    #print(chromosome)
    for n in chromosome:
        tmp = bin(n).replace("0b", "")
        tmp = tmp.zfill(8)
        sol_bin += tmp

    return sol_bin

def binary_to_decimal(pop):
    
    global population_in_decimal

    for slot,chromosome in enumerate(pop):
        playlist = []
        for index in range(0,56,8):
           playlist.append(int(chromosome[index:index+8], 2))
        population_in_decimal[slot] = playlist
    
    
def convert_pop_to_binary(pop):
    pop_bin = []

    for chromosome in pop:
        chromosome_bin = decimal_to_binary(chromosome)
        pop_bin.append(chromosome_bin)
    
    global population_in_binary
    population_in_binary = pop_bin
    return
  


###################### Calculating Average values of properties of song for a given activity#########################################
matrix = dataset.groupby('ACTIVITY').agg(
    sum_bpm=pd.NamedAgg(column='BPM', aggfunc=sum),
    sum_energy=pd.NamedAgg(column='ENERGY', aggfunc=sum),
    sum_danceability=pd.NamedAgg(column='DANCEABILITY', aggfunc=sum),
    sum_loudness=pd.NamedAgg(column='LOUDNESS', aggfunc=sum),
    sum_liveness=pd.NamedAgg(column='LIVENESS', aggfunc=sum),
    sum_valence=pd.NamedAgg(column='VALENCE', aggfunc=sum),
    sum_spch=pd.NamedAgg(column='SPCH ', aggfunc=sum),
    sum_pop=pd.NamedAgg(column='POP', aggfunc=sum),
)


averages = matrix.to_numpy()
countSeries = dataset['ACTIVITY'].value_counts()
totalCount = []
for i in range(7):
    totalCount.append(countSeries[i + 1])

for i in range(7):
    averages[i] = averages[i] / totalCount[i]


################################GENERATING INITIAL POPULATION##########################################


def gen_initial_population():
    pop_decimal = []
    for i in range(num_pop):
        songIDs = []
        for j in range(num_activity):
            num = random.randint(1, 256)
            songIDs.append(num)
        pop_decimal.append(songIDs)

    global population_in_decimal
    population_in_decimal = pop_decimal
    convert_pop_to_binary(population_in_decimal)
    return


######################## Calculate fitness_score ######################################################

def calculate_fitness(chromosome):
    for activity in range(num_activity):
        y = 0
        sid = chromosome[activity]
        for att in range(num_attributes):
            y += pow(song_att[sid - 1][att] - averages[activity][att], 2)

    return 100000 - y

def calculate_fitness_pop(pop):
    global fitness_score_pop
    tmp = []
    for chromosome in pop:
        sc = calculate_fitness(chromosome)
        tmp.append(sc)

    fitness_score_pop = tmp
    return


####################################################### SELECTING CHROMOMSOME ###########################################

def select_chromosome():
    if random.random() > 0.5:
        # Tournament Selection
        return tournament_selection()

    else:
        # Biased Roulette Selection
        return biased_roulette_selection()


def tournament_selection():
    p1 = random.randint(1, 10)
    p2 = random.randint(1, 10)

    if p1 == p2:
        return p1
    if fitness_score_pop[p1 - 1] > fitness_score_pop[p2 - 1]:
        return p1
    else:
        return p2


def biased_roulette_selection():
    # The probability of selection of a chromosome depends on the fitness score

    prob_distribution = []
    total_fitness_score = 0
    
    for sc in fitness_score_pop:
        total_fitness_score += sc
    
    for fi in fitness_score_pop:
        x = fi / total_fitness_score
        prob_distribution.append(x)

    a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random_parent = random.choices(a_list, prob_distribution)
    return random_parent[0]


def selection():
    mating_pool = []
    size_pool = int(0.6 * num_pop)

    for i in range(size_pool):
        selected_chromosome = select_chromosome()
        mating_pool.append(selected_chromosome)

    return mating_pool


def crossover(mating_pool):
   # crossover_site = random.randint(0, num_bits, 1)

    mother = random.choice(mating_pool)
    father = random.choice(mating_pool)
   
    for i in range(56):
        if( random.random() > 0.5 ):
           gene1 = mother[i]
           gene2 = father[i]
           mother = mother[0:i] + gene2 + mother[i+1:]
           father = father[0:i] + gene1 + father[i+1:]
    
    mating_pool.append(mother)
    mating_pool.append(father)
    
    return(mating_pool)


def mutation(mating_pool, mutation_rate):
    for offspring in mating_pool:
        for bit in range(56):
             if( random.random() > 0.5 ):
                  offspring = offspring[0:bit] + str(not offspring[bit]) + offspring[bit+1:]
            
    return(mating_pool)


def check_fitness_all():
    print("Check fitness")
    threshold = 1000 # Decide 
    for fit in fitness_score_pop:
        if fit > threshold :
            return True

    return False


def genetic_algorithm():
    global population_in_binary
    global population_in_decimal
    
    loop = 50
    while loop :
        # Selection of 60% of population using tournament selection and RW, randomly choosing b/w the two
        new_mating_pool = selection()
        new_mating_pool_binary = []
        for i in range(len(new_mating_pool)):
            new_mating_pool_binary.append(population_in_binary[new_mating_pool[i] -1])
       # print(len(new_mating_pool_binary[0]))
        
        
        # Reproduction
        crossover_rate = 0.2
        for count in range(int(crossover_rate*num_pop)):
           new_mating_pool_binary = crossover(new_mating_pool_binary)
        
        
        mutation_rate = 0.05
        New_pop = mutation(new_mating_pool_binary, mutation_rate)

        # Replacement
        population_in_binary =  New_pop
        
        # Recalculate fitness
        binary_to_decimal(population_in_binary)
        calculate_fitness_pop(population_in_decimal)
        #print(fitness_score_pop)
        y = min(fitness_score_pop)
        Y.append(y)
        loop -= 1
    return

def visualise():
    X = list(range(1,51))
    plt.plot(X, Y)
    plt.xlabel("Number Of Generations")
    plt.ylabel("Minimum Fitness Score")
    plt.show()
    return

###############################################GENETIC ALGORITHM BEGINS HERE###########################################

gen_initial_population()
calculate_fitness_pop(population_in_decimal)
genetic_algorithm()
visualise()








