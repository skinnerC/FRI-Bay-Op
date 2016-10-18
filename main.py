'''
References
http://www.sfu.ca/~ssurjano/branin.html
http://arxiv.org/pdf/1503.02946.pdf
http://deap.readthedocs.io/en/master/examples/pso_basic.html
'''
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multipolyfit as mpf
from deap import base, creator, tools
from scipy.interpolate.rbf import Rbf as rbf

methodCount = 0
random.seed(0)

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Particle', list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, pmin=None, pmax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    particle = creator.Particle(random.uniform(pmin[i], pmax[i]) for i in range(size))
            
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

    for i in range(len(particle)):
        new_pos = particle[i] + particle.speed[i]
        
        if new_pos > particle.pmax[i] or new_pos < particle.pmin[i]:
            particle.speed = [random.uniform(particle.smin, particle.smax) for _ in range(len(particle))]
            particle[:] = [random.uniform(particle.pmin[i], particle.pmax[i]) for i in range(len(particle))]
            return

    # Add speed to parameter and update particle
    particle[:] = list(map(operator.add, particle, particle.speed))

# Necessary if function does not take args as a list
def eval_wrapper(approx_eq, particle):
    return (approx_eq(*particle),)
    
def calc(approx_eq, particle):
    return (approx_eq(particle),)

def minimize(approx_eq, popsize, ngen, npar, smin, smax, pmin, pmax):
    toolbox = base.Toolbox()
    toolbox.register('particle', generate, size=npar, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register('population', tools.initRepeat, list, toolbox.particle)
    toolbox.register('update', update_particle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", eval_wrapper, approx_eq)

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
        for particleicle in pop:
            toolbox.update(particle, best=best)
#            Testing Code
#        if 0.387887 < best.fitness.values[0] < 0.407887:
#            return tuple([best] + [best.fitness.values[0]])
            
    return tuple([best] + [best.fitness.values[0]])

def generateSample():
    x1 = random.uniform(-5, 10)
    x2 = random.uniform(0, 15)
    return ([x1, x2], branin(x1, x2))

def Bayesian(samples, NPAR, PMIN = [-5, 0], PMAX=[10, 15], new_samples=20, graph=False):
    global methodCount
    methodCount = 0
    
    # Particle Swarm Optimization Parameters
    POPSIZE = 10#30 # Population size
    NGEN = 1000 # The number of gnerations 
    SMIN = -3 #-20 # Minimum speed value
    SMAX = 3 #20 # Maximum speed value
    
    for i in range(new_samples):
        # Seperate samples into xs and y arrays
        xs, y = map(np.array, zip(*samples))
        xs = np.array(xs)
        y = np.array(y)
            
#        # Creates a callable quadratic polyfit of the sample points
#        approx_eq = mpf.multipolyfit(xs, y, 3, model_out=True) # Where I change the diminsion
#        temp = np.zeros((xs.shape[0], xs.shape[1]+1))

        temp = []
        for i in range(NPAR):
            temp += [xs[:,i]]
        temp += [y]
        approx_eq = rbf(*temp)
        
        # Find minimum of approximate function
        min_xs =  minimize(approx_eq, POPSIZE, NGEN, NPAR, SMIN, SMAX, PMIN, PMAX)
        
        # Graph the aprox eq and min
        if graph:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            nx = np.arange(-5, 11)
            ny = np.arange(0, 16)
            X, Y = np.meshgrid(nx, ny)
            nz = np.array([approx_eq(nnx, nny) for nnx, nny in zip(np.ravel(X), np.ravel(Y))])
            Z = nz.reshape(X.shape)
            
            ax.plot_wireframe(X, Y, Z)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y')
        
            s_xy, s_z = zip(*samples)
            s_x, s_y = zip(*s_xy)
            ax.scatter(list(s_x), list(s_y), list(s_z), c='green')
            ax.scatter(min_xs[0][0], min_xs[0][1], min_xs[1], c='red')
            plt.show()        
        
        yield min_xs[0]
        new_y = yield
        samples += [(min_xs[0], new_y)]
        yield
    
    # Return the best xs from the sample ponts
    samples_xs, samples_y = zip(*samples)
    yield samples_xs[samples_y.index(min(samples_y))]
    return

# Global Minimum = 0.397887
# x1 = [-5, 10], x2 = [0, 15]
def branin(x1, x2, a = 1., b = 5.1/(4.*np.pi**2), c = 5./np.pi, r = 6., s = 10., t = 1./(8*np.pi)):
    global methodCount
    methodCount += 1
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)
    y = term1 + term2 + s
    return y

if __name__ == '__main__':
    eq = branin
    # Create 10 samlpe points and calculate their values
    samples = [generateSample() for _ in range(10)]
    
#    Graphing Code
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    nx = np.arange(-5, 11)
#    ny = np.arange(0, 16)
#    X, Y = np.meshgrid(nx, ny)
#    Z = branin(X, Y)
#    
#    ax.plot_wireframe(X, Y, Z)
#    ax.set_xlabel('x1')
#    ax.set_ylabel('x2')
#    ax.set_zlabel('y')
#        
#    s_xy, s_z = zip(*samples)
#    s_x, s_y = zip(*s_xy)
#    ax.scatter(list(s_x), list(s_y), list(s_z), c='green')
#    ax.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], [0.397887, 0.397887, 0.397887], c='red')
#    plt.show()    
    
    # Create generator
    bay = Bayesian(samples, 2)

    for i in range(20):
        xs = bay.next()
        bay.next()
        y = eq(*xs)
        bay.send(y)
    good_x =  bay.next()
    print good_x