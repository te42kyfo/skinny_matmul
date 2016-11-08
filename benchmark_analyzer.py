#!/bin/python

import sqlite3
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools as fnc

conn = sqlite3.connect('benchmarks.db')
conn.row_factory = sqlite3.Row

def makeGrid(res, axis1, axis2, value, max1=-1, max2=-1):
    if max1 < 0:
        max1 = fnc.reduce(max, map( lambda row: row[axis1], res))
    if max2 < 0:
        max2 = fnc.reduce(max, map( lambda row: row[axis2], res))

    grid = np.zeros((max1+1, max2+1))
    for row in res:
        if row[axis1] <= max1 and row[axis2] <= max2:
            grid[row[axis1]][row[axis2]] = row[value]
            if row['time'] < 0.0001 or not np.isfinite(row[value]):
                grid[row[axis1]][row[axis2]] = 0
    return grid

def fetchGrid(multype, device, types, name, inplace, zerobeta):
    cursor = conn.execute("SELECT * FROM benchmarks WHERE multype=? AND device=? and TYPES=?"
                          "AND name=? AND inplace=? AND zerobeta=?",
                          (multype, device, types, name, inplace, zerobeta))
    res = cursor.fetchall()
    return makeGrid(res, "N", "M", "bw")



def rooflinePlot(grid):
    fig=plt.figure( )
    imgplot = plt.imshow(grid / 144,
                         interpolation='nearest',
                         origin='lower',
                         vmin=0,
                         vmax=1,
                         cmap=plt.get_cmap('jet'))
    plt.suptitle("% of roofline performance")
    plt.xlabel("M")
    plt.ylabel("N")
    plt.xlim(0.5, plt.xlim()[1])
    plt.ylim(0.5, plt.ylim()[1])
    plt.colorbar()
    #plt.axes().set_aspect('auto')
    fig.tight_layout()
    plt.show()

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

def speedupPlot(grid1, grid2):

    
    grid1[grid1 < 0.2] = 1.0
    grid2[grid1 < 0.2] = 1.0
    grid1[grid2 < 0.2] = 1.0
    grid2[grid2 < 0.2] = 1.0
    diff = grid1/grid2

    diff = np.log10(diff)

    spread = max(abs(np.max(diff)), abs(np.min(diff)))
    print(spread)

    fig=plt.figure()
    imgplot = plt.imshow(diff,
                         interpolation='nearest',
                         origin='lower',
                         vmin = -spread, vmax=spread,
                         cmap=plt.get_cmap('bwr'))

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
            for dx in range(-2,3):
                for dy in range(-2,3):
                    if y+dy < grids[0].shape[0] and x+dx < grids[0].shape[1]:
                        for i in range(0, len(grids)):
                            histo[i] += grids[i][y+dy][x+dx]

            highest = 0
            value = 0
            for i in range(0, len(grids)):
                if histo[i] > value:
                    highest = i
                    value = histo[i]
            newBestGrid[y,x] = highest


    fig=plt.figure( )
    imgplot = plt.imshow(bestValue,
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



absolutePlot( fetchGrid("TSMTTSM", "Tesla K20m", "DR", "FGENV6", 0, 0))
absolutePlot( fetchGrid("TSMTTSM", "Tesla K20m", "DR", "FSMALL", 0, 0))





