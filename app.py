import json
import flask
import sqlite3
import math
import numpy as np

#Task4 - Import pyopencl here
import pyopencl as cl
import time
from flask import send_file

app = flask.Flask(__name__)
data = []
ctx  = []
queue = []


@app.route("/")
def index():
    return flask.render_template("index.html")

@app.route("/verlauf")
def verlauf():
    return send_file("templates/verlauf.jpg", mimetype='image/jpg')
#    return send_file("templates/mako.png", mimetype='image/jpg')


def gaussKernel(x, y, centerX, centerY, sigma):
    
    # Task 3 - Calculate the value for the current bin and observation for the naive CPU KDE
        
    return 0.0

@app.route("/data")
@app.route("/data/<int:numBins>/<string:minX>/<string:maxX>/<string:minY>/<string:maxY>")
def computeDataCPU(numBins=64,minX=-100,maxX=500,minY=-100,maxY=500):
    histogram = np.zeros((numBins , numBins), dtype = np.float)    
    #print(minY, maxY, minX, maxX)
    minArr = int(minY)
    maxArr = int(maxY)
    rangeArr = maxArr - minArr
    minDep = int(minX)
    maxDep = int(maxX)
    rangeDep = maxDep - minDep            
    maxBin = 0   
    

    # create the 2D histogram 
    # Task 3 - Remove the for loop
    for row in data: 
        if row[0] >= minDep and row[0] < maxDep and row[1] >= minArr and row[1] < maxArr: 
            x = math.floor(numBins * ((row[0] - minDep) / rangeDep))
            y = math.floor(numBins * ((row[1] - minArr) / rangeArr))
            histogram[y,x] += 1
            if maxBin < histogram[y,x]: 
                maxBin = histogram[y,x] 
                
    minX = int(minX)
    minY = int(minY)
    maxY = int(maxY)
    maxX = int(maxX)

    rangeX = maxX - minX
    rangeY = maxY - minY
    
    #Task 3 - Put the naive KDE implementation here  

    #Task 4 - Create the buffers, execute the OpenCL code and fetch the results 
                    
    return json.dumps({"minX": minDep, "maxX": maxDep, "minY": minArr, "maxY": maxArr, "maxBin": int(maxBin), "histogram": histogram.ravel().tolist()})

if __name__ == "__main__":
    import os

    port = 8002

    # Open a web browser pointing at the app.
    # os.system("open http://localhost:{0}".format(port))
    # perform db query 

    print("start db query")    
    conn = sqlite3.connect("data/flights.db")
    cursor = conn.cursor()

    sql = "SELECT DepDelay, ArrDelay FROM ontime WHERE TYPEOF(ArrDelay) IN ('integer', 'real') LIMIT 1000"
    # this query is slow at server start-up, but the histogram is quick to compute on the CPU
    #sql = "SELECT DepDelay, ArrDelay FROM ontime WHERE TYPEOF(ArrDelay) IN ('integer', 'real')  ORDER BY RANDOM() LIMIT 50000"
    # this query is quick, but just gives us the first 50000 items (remove the LIMITS 50000 to get all data points)
    # Task 3 - Lower the number of datapoints, set LIMIT to 10

    cursor.execute(sql)
    data = cursor.fetchall()

    print("data stored")

    #Task 4 - Setup and configure pyopencl

    # Set up the development server on port 8002.
    app.debug = False
    app.run(port = port)