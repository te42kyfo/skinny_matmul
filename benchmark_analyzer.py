#!/bin/python

import sqlite3
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools as fnc

conn = sqlite3.connect('benchmarks.db')
conn.row_factory = sqlite3.Row

def makeGrid(res, axis1, axis2, value):
    max1 = fnc.reduce(max, map( lambda row: row[axis1], res))
    max2 = fnc.reduce(max, map( lambda row: row[axis2], res))
    min1 = fnc.reduce(min, map( lambda row: row[axis1], res))
    min2 = fnc.reduce(min, map( lambda row: row[axis2], res))
    grid = np.ndarray((max1+1, max2+1))
    for row in res:
        grid[row[axis1]][row[axis2]] = row[value]
        if row['time'] < 0.0001:
            grid[row[axis1]][row[axis2]] = 0
    return grid

def fetchGrid(name,inplace, zerobeta):
    cursor = conn.execute("SELECT * FROM tsmm WHERE name=? AND inplace=? AND zerobeta=?",
                          (name, inplace, zerobeta))
    res = cursor.fetchall()
    return makeGrid(res, "N", "M", "bw")

def absolutePlot(grid):
    fig=plt.figure( )
    imgplot = plt.imshow(grid,
                         interpolation='nearest',
                         origin='lower',
                         cmap=plt.get_cmap('jet'))

    plt.xlim(0.5, plt.xlim()[1])
    plt.ylim(0.5, plt.ylim()[1])
    plt.colorbar()
    #plt.axes().set_aspect('auto')
    fig.tight_layout()
    plt.show()

def bestPlot(grids):

    bestValue = grids[0].flatten()
    bestGrid = np.zeros(bestValue.shape)

    gridNumber = 1
    for grid in grids[1:]:
        mask = bestValue < grid.ravel()
        bestGrid[mask] = gridNumber
        bestValue[mask] = grid.ravel()[mask]
        gridNumber += 1



    bestGrid = bestGrid.reshape(grids[0].shape)
    bestValue = bestValue.reshape(grids[0].shape)
    newBestGrid = bestGrid.copy()
    for y in range(1, bestGrid.shape[0]):
        for x in range(1, bestGrid.shape[1]):
            histo = [0] * len(grids)
            for dx in range(-1,2):
                for dy in range(-1,2):
                    if y+dy < grids[0].shape[0] and x+dx < grids[0].shape[1]:
                        histo[int(bestGrid[y+dy,x+dx])] += 1
            highest = bestGrid[y,x]
            count = histo[int(bestGrid[y,x])]
            for i in range(0, len(grids)):
                if( histo[i] > count):
                    highest = i
                    count = histo[i]
            newBestGrid[y,x] = highest


    fig=plt.figure( )
    imgplot = plt.imshow(newBestGrid,
                         interpolation='nearest',
                         origin='lower',
                         cmap=plt.get_cmap('jet'))


    plt.xlim(0.5, plt.xlim()[1])
    plt.ylim(0.5, plt.ylim()[1])
    plt.colorbar()
    plt.grid(True)
    #plt.axes().set_aspect('auto')
    fig.tight_layout()
    plt.show()

bestPlot([ fetchGrid( "VARIPG", 1, 1),
           fetchGrid( "CUBLAS", 1, 1),
           fetchGrid( "VARIP1", 1, 1),
           fetchGrid( "VARIP2", 1, 1),
           fetchGrid( "VARIP3", 1, 1)])







