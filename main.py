import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import sys
import math
from deap import base, creator, tools
from scipy.interpolate.rbf import Rbf as rbf

random.seed(0)

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Particle', list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, pmin=None, pmax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    particle = creator.Particle(np.random.uniform(pmin[i], pmax[i]) for i in range(size))
    particle.speed = [random.uniform(smin, smax) for _ in range(size)]
    particle.smin = smin
    particle.smax = smax
    particle.pmin = pmin
    particle.pmax = pmax
    return particle

def update_particle(particle, best, phi1, phi2):
    # Create new speeds
    u1 = (random.uniform(0, phi1) for _ in range(len(particle)))
    u2 = (random.uniform(0, phi2) for _ in range(len(particle)))
    v_u1 = map(operator.mul, u1, map(operator.sub, particle.best, particle))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, particle))
    particle.speed = list(map(operator.add, particle.speed, map(operator.add, v_u1, v_u2)))
    
    # Check if speeds are valid
    for i, speed in enumerate(particle.speed):
        if speed < particle.smin:
            particle.speed[i] = particle.smin
        elif speed > particle.smax:
            particle.speed[i] = particle.smax

    # Reset particle if its going to go out of bounds
    for i in range(len(particle)):
        new_pos = particle[i] + particle.speed[i]
        
        if new_pos > particle.pmax[i] or new_pos < particle.pmin[i]:
            particle.speed = [random.uniform(particle.smin, particle.smax) for _ in range(len(particle))]
            particle[:] = [random.uniform(particle.pmin[i], particle.pmax[i]) for i in range(len(particle))]
            return

    # Add speed to parameter and update particle
    particle[:] = list(map(operator.add, particle, particle.speed))

# Necessary if function does not take args as a list
def eval_wrapper(approx_eq, particle, samples):
    xs, y = map(np.array, zip(*samples))
    distances = xs - np.array(particle)
    s_hat = np.sqrt(np.sum(np.square(distances), axis=1))
    
    if min(s_hat) <= radius:
        return (sys.maxsize,)
        
    return (approx_eq(*particle),) # test with s_hat sum 
    
def calc(approx_eq, particle, samples):
    return (approx_eq(particle),)

def minimize(approx_eq, popsize, ngen, npar, smin, smax, pmin, pmax, samples=None):
    toolbox = base.Toolbox()
    toolbox.register('particle', generate, size=npar, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register('population', tools.initRepeat, list, toolbox.particle)
    toolbox.register('update', update_particle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", eval_wrapper, approx_eq, samples=samples)

    pop = toolbox.population(n = popsize)
    best = None

    for _ in range(ngen):
        for particle in pop:
            particle.fitness.values = toolbox.evaluate(particle)
            if not particle.best or particle.best.fitness > particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values
            if not best or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values
        for particle in pop:
            toolbox.update(particle, best=best)
            
#            Testing Code
#        print best.fitness.values
#        if -0.2 < best.fitness.values[0] < 0.2:
#            return tuple([best] + [best.fitness.values[0]])
    
    return tuple([best] + [best.fitness.values[0]])
    
def Bayesian(samples, NPAR, PMIN, PMAX, new_samples, graph=False):
    global methodCount
    global radius
    graph_num = 1
    methodCount = 0
    
    # Particle Swarm Optimization Parameters
    POPSIZE = 10 # Population size
    NGEN = 1000 # The number of gnerations 
    SMIN = -3 # Minimum speed value
    SMAX = 3 # Maximum speed value
    
    for i in range(new_samples):
        # Seperate samples into xs and y arrays
        xs, y = map(np.array, zip(*samples))
        xs = np.array(xs)
        y = np.array(y)
        
        # Create array of parameter arrays
        temp = []
        for j in range(NPAR):
            temp += [xs[:,j]]
        temp += [y]
        temp = np.array(temp)
        
        # Create RBF approximation
        approx_eq = rbf(*temp, function='gaussian')
        
        # Set radius value
        # Radius starts at a slightly less than the value of .2 * the shortest paramerter seach size
        # It then decreases proportionally such that the last 1/6 of the new new points are found greadily
        radius = (.22 - ((i+math.ceil(new_samples/5))*(.22/new_samples))) * max(np.array(PMAX) - np.array(PMIN))
        
        # Find minimum of approximate function
        min_xs =  minimize(approx_eq, POPSIZE, NGEN, NPAR, SMIN, SMAX, PMIN, PMAX, samples)
        
        # Graph the aprox eq and min
        if graph:
            fig = plt.figure()
            nx = np.arange(PMIN[0], PMAX[0]+1, 0.1)
            ny = np.arange(PMIN[1], PMAX[1]+1, 0.1)
            X, Y = np.meshgrid(nx, ny)
            Z = approx_eq(X, Y)
            s_xy, s_z = zip(*samples)
            s_x, s_y = zip(*s_xy)
            plt.contour(X, Y, Z, 15, linewidths=0.5, colors='k')
            plt.pcolormesh(X, Y, Z, cmap=plt.get_cmap('Greys'))
            plt.colorbar()
#            plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], s=20, c='red')
#            plt.scatter([1], [1], s=20, c='red')
#            plt.scatter([0.0898, -0.0898], [-0.7126, 0.7126], s=20, c='red')
            plt.scatter(s_x, s_y, s=30, marker='s', c='g', alpha='0.8')
            plt.scatter(min_xs[0][0], min_xs[0][1], s=30, marker='+', c='r')
            plt.xlim(PMIN[0], PMAX[0])
            plt.ylim(PMIN[1], PMAX[1])
            plt.xlabel('x1')
            plt.ylabel('x2')
            fig.suptitle('{0} Approximation'.format(savePath.split('/')[1]))
            plt.savefig('{0}Results-{1}.png'.format(savePath, graph_num))
            graph_num += 1
            plt.close(fig)
        
            # Forrester Graphs
#            fig = plt.figure()
#            nx = np.arange(PMIN[0], PMAX[0]+1, 0.01)
#            X = nx
#            Z = forrester(X)
#            s_xy, s_z = zip(*samples)
#            plt.plot(X, Z, label='Original')
#            plt.scatter([0.7572], [forrester(0.7572)], s=20, c='r')
#            Z = approx_eq(X)
#            plt.plot(X, Z, c='y', label='Approximation')
#            plt.scatter(s_xy, s_z, s=30, marker='s', c='g', alpha='0.8')
#            plt.scatter(min_xs[0], [forrester(min_xs[0][0])], s=30, marker='+', c='r')
#            plt.xlim(PMIN[0], PMAX[0])
#            plt.ylim(-10, 20)
#            plt.xlabel('x')
#            plt.ylabel('y')
#            fig.suptitle('{0}'.format('forrester'))
#            plt.legend(loc='upper center')
#            plt.savefig('{0}Results-{1}.png'.format(savePath, graph_num))
#            plt.close(fig)
#            graph_num += 1
        
        # Return parameters for user to try
        yield min_xs[0]
        
        # Get the y value of those parameters
        new_y = yield
        
        # Add that point to the samples
        samples += [(min_xs[0], new_y)]
        print 'New infill point: {0}'.format(str(samples[len(samples)-1]))
        
        # Keep the generator alligned
        yield
    
    fig = plt.figure()
    X = [h+1 for h in range(new_samples)]
    s_xy, s_z = zip(*samples)
    start = len(s_z)-new_samples
    Y = []
    for k in range(start, len(s_z)):
        Y += [s_z[k]]
    plt.plot(X, Y)
    plt.xlim(1, new_samples)
    plt.ylim(0, 20)
    plt.xlabel('Iteration')
    plt.ylabel('Value of New Sample Point')
    fig.suptitle('{0}'.format(savePath.split('/')[1]))
    plt.savefig('{0}Results.png'.format(savePath))
    plt.close(fig)
    
    # Return the best xs from the sample ponts
    samples_xs, samples_y = zip(*samples)
    yield (samples_xs[samples_y.index(min(samples_y))], min(samples_y))
    return

def generateSample(eq, PMIN, PMAX):
    xs = []
    for i in range(len(PMIN)):
        xs += [random.uniform(PMIN[i], PMAX[i])]
    return (xs, eq(*xs))

# Global Minimum = 0.397887
# x1 = [-5, 10], x2 = [0, 15]
def branin(x1, x2, a = 1., b = 5.1/(4.*np.pi**2), c = 5./np.pi, r = 6., s = 10., t = 1./(8*np.pi)):
    global methodCount
    methodCount += 1
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)
    y = term1 + term2 + s
    return y

# Global Minimum = 0
# xi = [-5, 10], [-2.048, 2.048]
def rosenbrock(x1, x2):
    global methodCount
    methodCount += 1
    a = 1. - x1
    b = x2 - x1*x1
    return a*a + b*b*100.

# Global Minimum = 
# x = [0, 1]
def forrester(x):
    global methodCount
    methodCount += 1
    return ((6*x - 2)**2)* np.sin(12*x - 4)

# x1 = [-3, 3], x2 = [-2, 2]
def six_hump_camel(x1, x2):
    global methodCount
    methodCount += 1
    return 4*(x1**2) - 2.1*(x1**4) + ((x1**6)/3.) + (x1*x2) - 4*(x2**2) + 4*(x2**4)
    
# xi = [-10, 10]
def colville(x1, x2, x3, x4):
    global methodCount
    methodCount += 1
    term1 = 100 * (x1**2-x2)**2;
    term2 = (x1-1)**2;
    term3 = (x3-1)**2;
    term4 = 90 * (x3**2-x4)**2;
    term5 = 10.1 * ((x2-1)**2 + (x4-1)**2);
    term6 = 19.8*(x2-1)*(x4-1);
    
    return term1 + term2 + term3 + term4 + term5 + term6;    
    
def rosenbrockD(xs, xd):
    sum = 0
    for i in range(1, xd):
        xi = xs[i-1]
        xnext = xs[i]
        new = 100*(xnext-(xi**2))**2 + (xi-1)**2
        sum += new
    return sum

def main():
    global methodCount # Might need (won't be used)
    
    # Min and max for each parameter (array of each)
    PMIN = [-2.048, -2.048, -2.048] # Need
    PMAX = [2.048, 2.048, 2.048] # Need
    
    def rosenbrock3D(x1, x2, x3): return rosenbrockD([x1, x2, x3], 3)
    
    eq = rosenbrock3D
    new_pts = 30 # Need
    graph_num = 0
    methodCount = 0 # Might need (won't be used)

#    #To use PSO on equation
#    minimize(eq, 10, 1000, 2, -3, 3, PMIN, PMAX)
#    print methodCount
    
    # Create 10 samlpe points and calculate their values
    # Samples need to be in the following form [([parameters], y), ([parameters], y), ...]
    samples = [generateSample(eq, PMIN, PMAX) for _ in range(15)] # Need (samples must be in the stated format)
    
    # Forrester Samples
#    samples = [([0.0], eq(0.0)), ([0.50], eq(0.50)), ([1.0], eq(1.0))]
    
    # Make folder
    if not os.path.exists(os.path.dirname(savePath)):
        try:
            os.makedirs(os.path.dirname(savePath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise    
    
    #Graphing Code
#    fig = plt.figure()
#    nx = np.arange(PMIN[0], PMAX[0]+1, 0.1)
#    ny = np.arange(PMIN[1], PMAX[1]+1, 0.1)
#    X, Y = np.meshgrid(nx, ny)
#    Z = eq(X, Y)
#        
#    s_xy, s_z = zip(*samples)
#    s_x, s_y = zip(*s_xy)
#    plt.contour(X, Y, Z, 15, linewidths=0.5, colors='k')
#    plt.pcolormesh(X, Y, Z, cmap=plt.get_cmap('Greys'))
#    plt.colorbar()
##    plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], s=20, c='red')
##    plt.scatter([1], [1], s=20, c='red')
##    plt.scatter([0.0898, -0.0898], [-0.7126, 0.7126], s=20, c='red')
#    plt.scatter(s_x, s_y, s=30, marker='s', c='g', alpha='0.8')
#    plt.xlim(PMIN[0], PMAX[0])
#    plt.ylim(PMIN[1], PMAX[1])
#    plt.xlabel('x1')
#    plt.ylabel('x2')
#    fig.suptitle('{0}'.format(eq.func_name))
#    plt.savefig('{0}Results-{1}.png'.format(savePath, graph_num))
#    plt.close(fig)
    
    # Create optimization generator
    bay = Bayesian(samples, 3, PMIN, PMAX, new_pts, graph=False) # Need
    
    # Create new_pts new sample points
    for i in range(new_pts): # Need
        # Get array of parameters that should be tried next
        xs = bay.next() # Need
        
        # Keeps the generator in sync
        bay.next() # Need
        
        # Get the value of the parameters that were sent and return that value to the generator
        # using send
        y = eq(*xs) # Need (this is where you would run the nn with the given parameters and get the fitness value)
        bay.send(y) # Need
    
    # Return the best set of parameters
    best_x =  bay.next() # Need
    print 'Final'
    print best_x
    
    file = open(savePath+'Results.txt', 'w')
    file.write(str(best_x[1])+'\n')
    file.write(str(best_x[0]))
    file.close()
    
if __name__ == '__main__':
    global savePath
    savePath = 'updated/rosenbrock3D/seed0/'
    
    for i in range(10):
        random.seed(i)
        savePath = savePath[:-2] + str(i) + '/'
        main()