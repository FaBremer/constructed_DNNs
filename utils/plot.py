from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

#Plotting functions

def plot_interp_2d(l,cheb,coefs,title="f(x,y)"):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    xxx = np.linspace(-1.0,1.0,l)
    X,Y = np.meshgrid(xxx,xxx)
    points = np.array([X.reshape(l*l,),Y.reshape(l*l,)]).T
    Z = cheb.interpolate(points,coefs)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')
    ax.plot_surface(X,Y,Z.reshape(l,l),rstride=1,cstride=1,cmap=cm.jet,edgecolor='black')
    fig = plt.figure(figsize=(8,8))
    plt.contour(X,Y,Z.reshape(l,l))

def plot_func_2d(l,f,title="f(x,y)",z=None):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    xxx = np.linspace(-1.0,1.0,l)
    X,Y = np.meshgrid(xxx,xxx)
    points = np.array([X.reshape(l*l,),Y.reshape(l*l,)]).T
    Z = f(points)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')
    if z is not None:
        x,y,z = z
        ax.scatter(x, y, z, marker='.',alpha=1)
    ax.plot_surface(X,Y,Z.reshape(l,l),rstride=1,cstride=1,cmap=cm.jet,edgecolor='black')
    fig = plt.figure(figsize=(8,8))
    plt.contour(X,Y,Z.reshape(l,l))

def generate_points(l):
    xxx = np.linspace(-1.0,1.0,l)
    X,Y = np.meshgrid(xxx,xxx)
    points = np.array([X.reshape(l*l,),Y.reshape(l*l,)]).T
    return points

def plot_points_2d(l,X,Y,Z,title="f(x,y)"):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')

    #ax.scatter(X, Y, Z, marker='^')
    ax.plot_surface(X,Y,Z.reshape(l,l),rstride=1,cstride=1,cmap=cm.jet,edgecolor='black')
    fig = plt.figure(figsize=(8,8))
    plt.contour(X,Y,Z.reshape(l,l))

def plot_net_2d(l,net,title="f(x,y)"):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    xxx = np.linspace(-1.0,1.0,l)
    X,Y = np.meshgrid(xxx,xxx)
    points = np.array([X.reshape(l*l,),Y.reshape(l*l,)]).T

    #points = np.matrix([X,Y]).T
    #points = np.array(points)
    Z = np.array([net.realize(p) for p in points])
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')

    #ax.scatter(X, Y, Z, marker='^')
    ax.plot_surface(X,Y,Z.reshape(l,l),rstride=1,cstride=1,cmap=cm.jet,edgecolor='black')
    fig = plt.figure(figsize=(8,8))
    plt.contour(X,Y,Z.reshape(l,l))