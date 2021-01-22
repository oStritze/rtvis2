import json
import flask
import sqlite3
import math
import numpy as np

#import scipy for now...
#from scipy.stats import norm

#Task4 - Import pyopencl here
import pyopencl as cl
import os
# for mac to use the gpu, use this to avoid being asked for devices:
os.environ["PYOPENCL_CTX"] = "1:1"
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1" 
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

@app.route("/viridis")
def verlauf1():
    return send_file("templates/viridis.jpg", mimetype='image/jpg')
@app.route("/mako")
def verlauf2():
    return send_file("templates/mako.jpg", mimetype='image/jpg')
@app.route("/magma")
def verlauf3():
    return send_file("templates/magma.jpg", mimetype='image/jpg')


def gaussKernel(x, y, centerX, centerY, sigma, amp=1):
    # Task 3 - Calculate the value for the current bin and observation for the naive CPU KDE
    val = 0.0
    ## using scipy
    # calculate the distance from center to point, that is the euclidian here
    #center = np.array((centerX, centerY))
    #point = np.array((x, y))
    #dist = np.linalg.norm(center-point) / sigma
    #val += norm.pdf(dist, 0, 1) 

    # use 2D gaussian function for the value instead of scipy
    xterm = (x-centerX)**2 / (2*sigma**2)
    yterm = (y-centerY)**2 / (2*sigma**2)
    val = amp * math.exp( -(xterm + yterm) )

    return val

@app.route("/data")
@app.route("/data/<int:numBins>/<string:minX>/<string:maxX>/<string:minY>/<string:maxY>/<int:sigma>/<string:runtime>")
def computeDataCPU(numBins=64,minX=-100,maxX=500,minY=-100,maxY=500, sigma=10, runtime="cpu"):
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
    """     
    for row in data: 
        if row[0] >= minDep and row[0] < maxDep and row[1] >= minArr and row[1] < maxArr: 
            x = math.floor(numBins * ((row[0] - minDep) / rangeDep))
            y = math.floor(numBins * ((row[1] - minArr) / rangeArr))
            histogram[y,x] += 1
            if maxBin < histogram[y,x]: 
                maxBin = histogram[y,x]  
    """

    minX = int(minX)
    minY = int(minY)
    maxY = int(maxY)
    maxX = int(maxX)

    rangeX = maxX - minX
    rangeY = maxY - minY

    #Task 3 - Put the naive KDE implementation here  
    _sigma = sigma # from function definition, default is 10
    n = len(data)

    if runtime == "cpu":
        start = time.time()
        stepX = math.floor(rangeX/numBins)
        stepY = math.floor(rangeY/numBins)
        for _x in range(numBins):
            for _y in range(numBins):
                thisX = ((minDep + _x*stepX) )
                thisY = ((minArr + _y*stepY) ) 
                for row in data:
                    val = gaussKernel(row[0], row[1], thisX, thisY, _sigma)
                    histogram[_y, _x] += val/_sigma

        histogram = histogram/n #as per formula, divide by n
        maxBin = np.max(histogram)
        end = time.time()
        print("CPU: ", end - start)
        return json.dumps({"minX": minDep, "maxX": maxDep, "minY": minArr, "maxY": maxArr, "maxVal": maxBin, "maxBin": int(maxBin), "histogram": histogram.ravel().tolist()})                

    
    #Task 4 - Create the buffers, execute the OpenCL code and fetch the results 
    if runtime == "gpu":
        start = time.time()

        nObs = len(data)
        xObs = np.array([row[0] for row in data], dtype=np.float32)
        yObs = np.array([row[1] for row in data], dtype=np.float32)
        
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags

        g_xObs = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xObs)
        g_yObs = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yObs)

        prg = cl.Program(ctx, """
        float isotropicGaussKernel2DCL(float x, float y, float center_x, float center_y, float sigma)
        {   
            int amp = 1;
            float xterm = pow(x - center_x, 2) / (2*pow(sigma, 2));
            float yterm = pow(y - center_y, 2) / (2*pow(sigma, 2));
            float val = amp * exp( -(xterm + yterm) );
            return val;
        }

        __kernel void DensityEstimation(__global float *observationsX, __global float *observationsY,
            unsigned long numObservations, float sigma,
            float minX, float maxX, float minY, float maxY,
            long numBins, __global float *kde_image) {
                int index = get_global_id(0); // same as blockIdx.x * blockDim.x + threadIdx.x apparently...

                float rangeX = maxX - minX;
                float rangeY = maxY - minY;
                int stepX = floor(rangeX/numBins);
                int stepY = floor(rangeY/numBins);

                for (int _y = 0; _y < numBins; _y++){
                    float thisY = ( (minY + _y*stepY) );
                    for (int _x = 0; _x < numBins; _x++){
                        long double this_kde = 0.0;
                        float thisX = ( (minX + _x*stepX) );
                        for (int n = 0; n < numObservations; n++){
                                this_kde += ( isotropicGaussKernel2DCL(observationsX[n], observationsY[n],
                                        thisX, thisY, sigma) /sigma );
                        }
                        kde_image[index + _y*64 + _x] = this_kde;
                    }
                    //kde_image[index + j*64 ] += this_kde;
                    //kde_image[index + j*_y] += this_kde; // wild 2d shit that may be in the wrong order
                    //kde_image[index + j] = this_kde; // only one row, all cols
                    //kde_image[index] = this_kde; // nothing
                }
                //kde_image[100] = 10;
                //kde_image[_y] = 10;
                //kde_image = kde_image/float(numObservations);
        }
        """).build()

        res_kde = np.zeros((numBins,numBins), dtype=np.float32)
        g_kde_image = cl.Buffer(ctx, mf.WRITE_ONLY, res_kde.nbytes)
        #prg.DensityEstimation( queue, res_kde.shape, None,
        prg.DensityEstimation( queue, (1, 1), None,
            g_xObs, g_yObs, np.int32(nObs), np.float32(_sigma),
            np.float32(minX), np.float32(maxX), np.float32(minY), np.float32(maxY),
            np.int32(numBins), g_kde_image )
        kde_image = np.empty_like(res_kde)
        cl.enqueue_copy(queue, kde_image, g_kde_image)
        kde_image = np.nan_to_num(kde_image)
        kde_image = kde_image /nObs #as per formula, divide by n
        maxBin = float(np.max(kde_image))

        end = time.time()
        print("GPU: ", end - start)
        return json.dumps({"minX": minDep, "maxX": maxDep, "minY": minArr, "maxY": maxArr, "maxVal": maxBin, "maxBin": int(maxBin), "histogram": kde_image.ravel().tolist()})                

    """ 
    # DEBUG
    print(xObs)
    # print(yObs)
    # print(kde_image.shape)
    print(kde_image[0])
    #print(histogram.shape)
    print(histogram[0])
    print(np.max(histogram))
    print(maxBin)
    """

    #return json.dumps({"minX": minDep, "maxX": maxDep, "minY": minArr, "maxY": maxArr, "maxBin": int(maxBin), "histogram": histogram.ravel().tolist()})

if __name__ == "__main__":
    import os

    port = 8002

    # Open a web browser pointing at the app.
    # os.system("open http://localhost:{0}".format(port))
    # perform db query 

    print("start db query")    
    conn = sqlite3.connect("data/flights.db")
    cursor = conn.cursor()

    sql = "SELECT DepDelay, ArrDelay FROM ontime WHERE TYPEOF(ArrDelay) IN ('integer', 'real') LIMIT 100"
    # this query is slow at server start-up, but the histogram is quick to compute on the CPU
    #sql = "SELECT DepDelay, ArrDelay FROM ontime WHERE TYPEOF(ArrDelay) IN ('integer', 'real')  ORDER BY RANDOM() LIMIT 50000"
    # this query is quick, but just gives us the first 50000 items (remove the LIMITS 50000 to get all data points)
    # Task 3 - Lower the number of datapoints, set LIMIT to 10
    #sql = "SELECT DepDelay, ArrDelay FROM ontime WHERE TYPEOF(ArrDelay) IN ('integer', 'real')  ORDER BY RANDOM() LIMIT 400"

    cursor.execute(sql)
    data = cursor.fetchall()

    print("data stored")

    #Task 4 - Setup and configure pyopencl

    # Set up the development server on port 8002.
    app.debug = False
    app.run(port = port)