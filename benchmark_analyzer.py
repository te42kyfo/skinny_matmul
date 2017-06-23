#!/bin/python

import sqlite3
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools as fnc

conn = sqlite3.connect('benchmarks.db')
conn.row_factory = sqlite3.Row

arithIntensity = np.zeros((100, 100))
for m in range(1,100):
    for n in range(1,100):
        arithIntensity[m, n] = 2*m*n/(m+n)


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

def fetchGrid(multype, device, types, name, inplace, zerobeta, value):
    cursor = conn.execute("SELECT * FROM benchmarks WHERE multype=? AND device=? and TYPES=?"
                          "AND name=? AND inplace=? AND zerobeta=?",
                          (multype, device, types, name, inplace, zerobeta))
    res = cursor.fetchall()
    if len(res) == 0:
        print("No Results for " + multype + " " + device + " " + name + " " + str(inplace) + " " + str(zerobeta))
        return np.zeros((0,0))
    return makeGrid(res, "N", "M", value)

def fetchGridWhere(whereClause):
    cursor = conn.execute("SELECT * FROM benchmarks WHERE " + whereClause)
    res = cursor.fetchall()
    if len(res) == 0:
        print("No Results for " + whereClause)
        return np.zeros((0,0))
    return makeGrid(res, "N", "M", "bw", 64, 64)



def rooflinePlot(multype, device, types, name, inplace, zerobeta, streamBW, peakFP):
    bwgrid = fetchGrid(multype, device, types, name, inplace, zerobeta, "bw")
    flopgrid = fetchGrid(multype, device, types, name, inplace, zerobeta, "flops")
    fig=plt.figure( )
    imgplot = plt.imshow( np.maximum(bwgrid / streamBW, flopgrid / peakFP),
                         interpolation='nearest',
                         origin='lower',
                         vmin=0,
                         vmax=1,
                         cmap=plt.get_cmap('jet'))
    plt.suptitle(device + ", " + name + ", " + types + ", % of roofline performance",size=22)
    plt.xlabel("M")
    plt.ylabel("N")
    plt.xlim(0.5, plt.xlim()[1])
    plt.ylim(0.5, plt.ylim()[1])
    plt.colorbar()
    #plt.axes().set_aspect('auto')
#    fig.tight_layout()
    plt.savefig("roofline_" + device.replace(" ", "_") + "_" + types + "_" + name +".png", transparent=True)
    plt.show()
    plt.close()


def absolutePlot(multype, device, types, name, inplace, zerobeta, value):
    grid = fetchGrid(multype, device, types, name, inplace, zerobeta, value)
    fig=plt.figure( )
    imgplot = plt.imshow(grid,
                         interpolation='nearest',
                         origin='lower',
                         cmap=plt.get_cmap('jet'))
    plt.suptitle(device + ", " + name + ", " + types + ", " + value,size=22)
    plt.xlabel("M")
    plt.ylabel("N")
    plt.xlim(0.5, plt.xlim()[1])
    plt.ylim(0.5, plt.ylim()[1])
    plt.colorbar()
    #plt.axes().set_aspect('auto')
    #fig.tight_layout()
    name = value + "_" + device.replace(" ", "_") + "_" + types + "_" + name +".png"
    plt.savefig(name, transparent=True, bbox_inchsasdes="tight")
    return name

def speedupPlot(grid1, grid2, max1 = -1, max2 = -2):
    if(max1 > 0 and max2 > 0):
        grid1 = grid1[0:max1,0:max2];
        grid2 = grid2[0:max1,0:max2];

    grid1[grid1 < 0.2] = 1.0
    grid2[grid1 < 0.2] = 1.0
    grid1[grid2 < 0.2] = 1.0
    grid2[grid2 < 0.2] = 1.0
    diff = grid1/grid2

    diff = np.log2(diff)

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
    plt.savefig("speedup_.png")
    #fig.tight_layout()
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
    imgplot = plt.imshow(bestGrid,
                         interpolation='nearest',
                         origin='lower',
                         cmap=plt.get_cmap('jet'))


    plt.xlim(0.5, plt.xlim()[1])
    plt.ylim(0.5, plt.ylim()[1])
    plt.colorbar()
    plt.grid(True)
    #plt.axes().set_aspect('auto')
    #fig.tight_layout()
    plt.show()




def analyseGENV3X():
    cursor = conn.execute("SELECT * FROM benchmarks WHERE multype=? AND device=? and TYPES=?"
                          "AND name=? AND inplace=? AND zerobeta=? AND time > 0 "
                          "AND usr1_name=?",
                          ("TSMTTSM",  "Tesla K20m", "DR", "GENV3X", 0, 0, "threads_per_n"))
    res = cursor.fetchall()

    n_per_thread = []
    values = []
    colors = []
    for row in res:
        #    print( str(row['N']) + " " + str(row['bw']) + " " + row['usr1_val'])
        n_per_thread.append(  row['N'] )
        values.append( row['bw'] )
        colors.append( min(row['N'] / (int(row['usr1_val']) +1), 6))

        plt.scatter( n_per_thread, values, s=200, c=colors, alpha=0.8 )
        plt.show()

        grids = []
        for i in range(0, 16):
            grids.append( fetchGridWhere("multype=\"TSMTTSM\" AND device=\"Tesla K20m\" and TYPES=\"DR\" AND name=\"GENV3X\" AND inplace=0 AND zerobeta=0 AND usr1_name=\"threads_per_n\" AND usr1_val=\"" + str(i) + "\""))

        bestPlot(grids)


