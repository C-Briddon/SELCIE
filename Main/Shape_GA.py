#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:25:57 2021

@author: ppycb3

Environment - fenics2019

DEAP Genetic Algoithm (GA) code.
"""
import timeit
import random
import numpy as np
import matplotlib.pyplot as plt
import operator
from Meshing_Tools import Meshing_Tools
from deap import gp, base, creator, tools, algorithms
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class Shape_GA():
    def __init__(self, fitness_function, Np = 100, tournament_size = 10, initial_tree_depth_min = 6, 
                 initial_tree_depth_max = 6, MAX_HEIGHT = 6):
        
        self.Np = Np
        self.initial_tree_depth_min = initial_tree_depth_min
        self.initial_tree_depth_max = initial_tree_depth_max
        self.MAX_HEIGHT = MAX_HEIGHT
        
        self.prob_X = 0.8
        self.prob_M = 0.1
        
        self.generation_max = 100
        self.stagnation_max = 20
        
        self.best_fits = []
        self.T, self.Mean, self.Max, self.Min, self.STD = [], [], [], [], []
        
        o = Meshing_Tools()
        
        self.toolbox = base.Toolbox()
        self.pset = gp.PrimitiveSetTyped("Main", [], list)
        self.pset.addPrimitive(o.apply_add, [list, list], list, "add")
        self.pset.addPrimitive(o.apply_sub, [list, list], list, "sub")
        self.pset.addPrimitive(o.apply_inx, [list, list], list, "inx")
        #self.pset.addPrimitive(o.apply_ninx, [list, list], list, "ninx")
        self.pset.addPrimitive(o.apply_rtz, [list, float], list, "rot")
        self.pset.addPrimitive(o.apply_tlx, [list, float], list, "tlx")
        self.pset.addPrimitive(o.apply_tly, [list, float], list, "tly")
        self.pset.addPrimitive(o.unity, [float], float, "I")
        self.pset.addEphemeralConstant('%f' % timeit.default_timer() , lambda: random.uniform(-1.0, 1.0), float)
        self.pset.addPrimitive(o.apply_create_disk, [float, float], list, "C_Disk")
        self.pset.addPrimitive(o.apply_create_rectangle, [float, float], list, "C_Rect")
        self.pset.addTerminal([], list, "empty")
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,pset=self.pset)
        
        self.toolbox.register("expr", gp.genGrow, pset = self.pset, 
                              min_ = self.initial_tree_depth_min, 
                              max_ = self.initial_tree_depth_max)
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", fitness_function , pset = self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize = tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), self.MAX_HEIGHT))
        self.toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), self.MAX_HEIGHT))
        
        
        stats_size = tools.Statistics(key=operator.attrgetter('height'))
        stats_size.register("avg", np.mean)
        stats_size.register("min", min)
        stats_size.register("max", max)
        
        return None
    
    
    def run_GA(self):
        
        'Create initial population and evaluate its fitness.'
        pop = self.toolbox.population(n=self.Np)
        hof = tools.HallOfFame(1)
        
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        hof.update(population = pop)
        best_fit = hof[0].fitness.values[0]
        
        g = 0
        stagnate = 0
        while g < self.generation_max:
            
            # A new generation
            t0 = timeit.default_timer()
            g += 1
            print("-- Generation %i --" % g)
            
            # Select the next generation individuals and make clones.
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Use algorthm so decorated versions of functions are used so limits are applied.
            offspring = algorithms.varAnd(offspring, self.toolbox, self.prob_X, self.prob_M)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace ond population with new.
            pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            
            t1 = timeit.default_timer()
            
            hof.update(population = pop)
            self.save_fitness(fits, t1 - t0)
            
            
            # Update termination condition.
            if best_fit == hof[0].fitness.values[0]:
                stagnate += 1
            else:
                best_fit = hof[0].fitness.values[0]
                stagnate = 0
            
            if stagnate > self.stagnation_max:
                print("Genetic Algorithm has converged.")
                break
        
        
        self.plot_results()
        self.plot_tree(hof[0])
        
        return gp.compile(hof[0], self.pset)
    
    
    def save_fitness(self, fits, dt):
        length = len(fits)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
        self.T.append(dt)
        self.Min.append(min(fits))
        self.Max.append(max(fits))
        self.Mean.append(mean)
        self.STD.append(std)
        
        if self.best_fits:
            if self.best_fits[-1] > self.Min[-1]:
                self.best_fits.append(self.Min[-1])
            else:
                self.best_fits.append(self.best_fits[-1])
        else:
            self.best_fits.append(self.Min[-1])
        
        print('T =', self.T[-1])
        
        return None
    
    
    def plot_results(self):
        'Plot time results.'
        x = list(range(1,len(self.T)+1))
        plt.figure()
        plt.ylabel('T')
        plt.xlabel('I')
        plt.plot(x,self.T)
        
        print('Total time is:', sum(self.T))
        
        'Plot fitness against time.'
        plt.figure()
        plt.ylabel('fitness')
        plt.xlabel('I')
        plt.errorbar(x, self.Mean, yerr = self.STD)
        plt.plot(x, self.Max, 'r-')
        plt.plot(x, self.Min, 'r-')
        
        plt.figure()
        plt.ylabel('fitness min')
        plt.xlabel('I')
        plt.yscale('log')
        plt.plot(x, self.Min, 'r--')
        plt.plot(x, self.best_fits, 'k-')
        
        return None
    
    
    def plot_tree(self, expr):
        plt.figure()
        nodes, edges, labels = gp.graph(expr)
        
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g, prog="dot")
        
        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()
        return None