#!/usr/bin/env python3
from abc import ABC, abstractmethod # Abstract Base Class
# import cgi
# import cgitb
import sys # To access stdout
import random as rand # For random number generation
import math # For sin, cos, pi and log functions
from dataclasses import dataclass # To create data classes
import os
import bz2 # For compressing the output to BZ2
import io
from shapely.wkt import loads
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import geopandas as gpd
from shapely.ops import unary_union
from scipy import sparse
import time
from joblib import parallel_backend
from shapely.geometry import box
import pandas as pd
import networkx as nx
from scipy import stats, interpolate
import ctypes
import subprocess
import json
# from hilbertcurve.hilbertcurve import HilbertCurve

# Genreate a random number in the range [min, max)
def uniform(min, max):
    return rand.random() * (max - min) + min

# Generate a random number from a Bernoulli distribution
def bernoulli(p):
    return 1 if rand.random() < p else 0

# Generate a random number from a normal distribution with the given mean and standard deviation
def normal(mu, sigma):
    return mu + sigma * math.sqrt(-2 * math.log(rand.random())) * math.sin(2 * math.pi * rand.random())

# Generate a random integer number in the range [1, n]
def dice(n):
    return math.floor(rand.random() * n) + 1

# A class that accepts geometries and writes them to the appropriate output
class DataSink(ABC):
    @abstractmethod
    def writePoint(self, coordinates):
        pass

    @abstractmethod
    def writeBox(self, coordinates):
        pass
#add one for polygon
    @abstractmethod
    def writePolygon(self, coordinates):
        pass
    @abstractmethod
    def flush(self):
        pass

class CSVSink(DataSink):
    def __init__(self, output):
        self.output = output
    
    def writePoint(self, coordinates):
        self.output.write(",".join([str(elem) for elem in coordinates]))
        self.output.write("\n")

    def writeBox(self, minCoordinates, maxCoordinates):
        self.output.write(",".join([str(elem) for elem in minCoordinates]))
        self.output.write(",")
        self.output.write(",".join([str(elem) for elem in maxCoordinates]))
        self.output.write("\n")
    #add one for polygon
    def writePolygon(self, coordinates):
        for coord in coordinates:
            self.output.write(",".join([str(elem) for elem in coord]))
            self.output.write(";")
        self.output.write(f"{coordinates[0][0]},{coordinates[0][1]}")
        self.output.write("\n")
    def flush(self):
        self.output.flush()

class WKTSink(DataSink):
    def __init__(self, output):
        self.output = output
    
    def writePoint(self, coordinates):
        self.output.write("POINT(")
        self.output.write(" ".join([str(elem) for elem in coordinates]))
        self.output.write(")\n")

    def writeBox(self, minCoordinates, maxCoordinates):
        self.output.write("POLYGON((")
        self.output.write(f"{minCoordinates[0]} {minCoordinates[1]},")
        self.output.write(f"{maxCoordinates[0]} {minCoordinates[1]},")
        self.output.write(f"{maxCoordinates[0]} {maxCoordinates[1]},")
        self.output.write(f"{minCoordinates[0]} {maxCoordinates[1]},")
        self.output.write(f"{minCoordinates[0]} {minCoordinates[1]}")
        self.output.write("))\n")
    def writePolygon(self, coordinates):
        self.output.write("POLYGON((")
        for coord in coordinates:
            self.output.write(" ".join([str(elem) for elem in coord]))
            self.output.write(",")
        self.output.write(f"{coordinates[0][0]} {coordinates[0][1]}")
        self.output.write("))\n")

    def flush(self):
        self.output.flush()

class GeoJSONSink(DataSink):
    def __init__(self, output):
        self.output = output
        self.first_record = True
        self.output.write('{"type": "FeatureCollection", "features": [')
    
    def writePoint(self, coordinates):
        if not self.first_record:
            self.output.write(",")
        self.output.write("\n")
        self.output.write('{"type": "Feature", "geometry": { "type": "Point", "coordinates": [')
        self.output.write(",".join([str(elem) for elem in coordinates]))
        self.output.write("]} }")
        self.first_record = False
        

    def writeBox(self, minCoordinates, maxCoordinates):
        if not self.first_record:
            self.output.write(",")
        self.output.write("\n")
        self.output.write('{"type": "Feature", "geometry": { "type": "Polygon", "coordinates": [[')
        self.output.write(f"[{minCoordinates[0]},{minCoordinates[1]}],")
        self.output.write(f"[{maxCoordinates[0]},{minCoordinates[1]}],")
        self.output.write(f"[{maxCoordinates[0]},{maxCoordinates[1]}],")
        self.output.write(f"[{minCoordinates[0]},{maxCoordinates[1]}],")
        self.output.write(f"[{minCoordinates[0]},{minCoordinates[1]}]")
        self.output.write("]]} }")
        self.first_record = False
    def writePolygon(self, coordinates):
        if not self.first_record:
            self.output.write(",")
        self.output.write("\n")
        self.output.write('{"type": "Feature", "geometry": { "type": "Polygon", "coordinates": [[')
        #array for coordinates
        temp1 = []
        for coord in coordinates:
            #joins each of the array elements in array coordinates into a string separated by a comma
            coords = ",".join([str(elem) for elem in coord])
            out = f"[{coords}]"
            #appends formatted string into the array
            temp1.append(out)
        #formats first array in array coordinates to append at the end
        lastCoord= ",".join([str(elem) for elem in coordinates[0]])
        temp1.append(f"[{lastCoord}]")
        self.output.write(",".join(temp1))
        self.output.write("]]} }")
        self.first_record = False

        #self.output.write(" }")
    def flush(self):
        self.output.write("]}")
        self.output.flush()

# A data sink that takes points and converts them to boxes
class PointToBoxSink(DataSink):
    def __init__(self, sink, maxsize):
        self.sink = sink
        self.maxsize = maxsize
    
    def writePoint(self, coordinates):
        # Generate a box around the coordinates
        minCoordinates = []
        maxCoordinates = []
        for d in range(0, len(coordinates)):
            size = uniform(0, self.maxsize[d])
            minCoordinates.append(coordinates[d] - size)
            maxCoordinates.append(coordinates[d] + size)
        self.writeBox(minCoordinates, maxCoordinates)
    
    def writeBox(self, minCoordinates, maxCoordinates):
        self.sink.writeBox(minCoordinates, maxCoordinates)
    
    def writePolygon(self, coordinates):
        sys.stdout.write("writing a polygon")
        #google what join does
    def flush(self):
        self.sink.flush()
class PointToPolygonSink(DataSink):
    def __init__(self, sink, maxseg, polysize):
        self.sink = sink
        self.maxseg = maxseg
        self.polysize = polysize
    def transform(self, center, angle):
        x = center[0] + self.polysize * math.cos(angle)
        y = center[1] + self.polysize * math.sin(angle)
        return [x, y]
    def flush(self):
        self.sink.flush()

    def writePoint(self, coordinates):
        center = coordinates
        minSegs = 3
        if(self.maxseg <= 3):
            numSegments = minSegs
        else:
            numSegments = dice(self.maxseg - minSegs) + minSegs
        angles = []
        for increment in range(0, numSegments):
            angles.append(uniform(0, math.pi * 2))
        angles.sort()
        points = []
        for angle in angles:
            points.append(self.transform(center, angle))
        self.writePolygon(points)
    
    def writeBox(self, coordinates):
        print("got to write box")
    
    def writePolygon(self, coordinates):
        self.sink.writePolygon(coordinates)

class BZ2OutputStream:
    def __init__(self, output):
        self.output = output
        self.compressor = bz2.BZ2Compressor()
    
    def write(self, data):
        compressedData = self.compressor.compress(bytes(data, "utf-8"))
        self.output.write(compressedData)
    
    def flush(self):
        data = self.compressor.flush() # Get the last bits of data remaining in the compressor
        self.output.write(data)
        self.output.flush()

# A data sink that converts all shapes using an affine transformation before writing them
class AffineTransformSink(DataSink):
    def __init__(self, sink, dim, affineMatrix):
        self.sink = sink
        assert len(affineMatrix) == dim * (dim + 1)
        squarematrix = []
        for d in range(0, dim):
            squarematrix.append(affineMatrix[d * (dim+1) : (d+1) * (dim+1)])
        squarematrix.append([0]* dim + [1])
        self.affineMatrix = squarematrix
    
    # Transform a point using an affine transformation matrix
    def affineTransformPoint(self, coordinates):
        # Transform the array

        # The next line uses numpy for matrix multiplication. But we use our own code to reduce the dependency
        # Append [1] to the input cordinates to match the affine transformation
        # Remove the last element from the result
        # transformed = np.matmul(self.affineMatrix, coordinates + [1])[:-1]

        # Matrix multiplication using a regular code
        dim = len(coordinates)
        transformed = [0] * dim
        for i in range(0, dim):
            transformed[i] = self.affineMatrix[i][dim]
            for d in range(0, dim):
                transformed[i] += coordinates[d] * self.affineMatrix[i][d]

        return transformed
    
    def writePoint(self, coordinates):
        self.sink.writePoint(self.affineTransformPoint(coordinates))

    def writeBox(self, minCoordinates, maxCoordinates):
        self.sink.writeBox(self.affineTransformPoint(minCoordinates), self.affineTransformPoint(maxCoordinates))
    
    def writePolygon(self, coordinates):
        for coord in coordinates:
            self.sink.writeBox(self.affineTransformPoint(coord[0]), self.affineTransformPoint(coord[1]))

    def flush(self):
        self.sink.flush()

# An abstract generator
class Generator(ABC):
    
    def __init__(self, card, dim):
        self.card = card
        self.dim = dim

    # Set the sink to which generated records will be written
    def setSink(self, datasink):
        self.datasink = datasink

    # Check if the given point is valid, i.e., all coordinates in the range [0, 1]
    def isValidPoint(self, point):
        for x in point:
            if not (0 <= x <= 1):
                return False
        return True

    # Generate all points and write them to the data sink
    @abstractmethod
    def generate(self):
        pass

class PointGenerator(Generator):
    def __init__(self, card, dim):
        super(PointGenerator, self).__init__(card, dim)
    
    @abstractmethod
    def generatePoint(self, i, prevpoint):
        pass

    def generate(self):
        i = 0
        prevpoint = None
        while i < self.card:
            newpoint = self.generatePoint(i, prevpoint)
            if self.isValidPoint(newpoint):
                self.datasink.writePoint(newpoint)
                prevpoint = newpoint
                i += 1
        self.datasink.flush()

# Generate uniformly distributed points
class UniformGenerator(PointGenerator):

    def __init__(self, card, dim):
        super(UniformGenerator, self).__init__(card, dim)

    def generatePoint(self, i, prev_point):
        return [rand.random() for d in range(self.dim)]

# Generate points from a diagonal distribution
class DiagonalGenerator(PointGenerator):

    def __init__(self, card, dim, percentage, buffer):
        super(DiagonalGenerator, self).__init__(card, dim)
        self.percentage = percentage
        self.buffer = buffer

    def generatePoint(self, i, prev_point):
        if bernoulli(self.percentage) == 1:
            return [rand.random()] * self.dim
        else:
            c = rand.random()
            d = normal(0, self.buffer / 5)
            return [(c + (1 - 2 * (x % 2)) * d / math.sqrt(2)) for x in range(self.dim)]

class GaussianGenerator(PointGenerator):
    def __init__(self, card, dim):
        super(GaussianGenerator, self).__init__(card, dim)

    def generatePoint(self, i, prev_point):
        return [normal(0.5, 0.1) for d in range(self.dim)]

class SierpinskiGenerator(PointGenerator):
    def __init__(self, card, dim):
        super(SierpinskiGenerator, self).__init__(card, dim)

    def generatePoint(self, i, prev_point):
        if i == 0:
            return [0.0, 0.0]
        elif i == 1:
            return [1.0, 0.0]
        elif i == 2:
            return [0.5, math.sqrt(3) / 2]
        else:
            d = dice(5)

            if d == 1 or d == 2:
                return self.get_middle_point(prev_point, [0.0, 0.0])
            elif d == 3 or d == 4:
                return self.get_middle_point(prev_point, [1.0, 0.0])
            else:
                return self.get_middle_point(prev_point, [0.5, math.sqrt(3) / 2])

    def get_middle_point(self, point1, point2):
        middle_point_coords = []
        for i in range(len(point1)):
            middle_point_coords.append((point1[i] + point2[i]) / 2)
        return middle_point_coords

class BitGenerator(PointGenerator):
    def __init__(self, card, dim, prob, digits):
        super(BitGenerator, self).__init__(card, dim)
        self.prob = prob
        self.digits = digits

    def generatePoint(self, i, prev_point):
        return [self.bit() for d in range(self.dim)]

    def bit(self):
        num = 0.0
        for i in range(1, self.digits + 1):
            c = bernoulli(self.prob)
            num = num + c / (math.pow(2, i))
        return num

# A two-dimensional box with depth field. Used with the parcel generator
@dataclass
class BoxWithDepth:
    depth: int
    x: float
    y: float
    w: float
    h: float

class ParcelGenerator(Generator):
    def __init__(self, card, dim, split_range, dither):
        super(ParcelGenerator, self).__init__(card, dim)
        self.split_range = split_range
        self.dither = dither

    def generate(self):
        # Using dataclass to create BoxWithDepth, which stores depth of each box in the tree
        # Depth is used to determine at which level to stop splitting and start printing    
        box = BoxWithDepth(0, 0.0, 0.0, 1.0, 1.0)
        boxes = [] # Empty stack for depth-first generation of boxes
        boxes.append(box)
        
        max_height = math.ceil(math.log(self.card, 2))
        
        # We will print some boxes at last level and the remaining at the second to last level 
        # Number of boxes to split on the second to last level
        numToSplit = self.card - pow(2, max(max_height - 1, 0))
        numSplit = 0
        boxes_generated = 0

        while boxes_generated < self.card:
            b = boxes.pop()
            
            if b.depth >= max_height - 1:
                if numSplit < numToSplit: # Split at second to last level and print the new boxes
                    b1, b2 = self.split(b, boxes)
                    numSplit += 1
                    self.dither_and_print(b1)
                    self.dither_and_print(b2)
                    boxes_generated += 2
                else: # Print remaining boxes from the second to last level 
                    self.dither_and_print(b)
                    boxes_generated += 1
                    if boxes_generated == 10: # Early flush to ensure immediate download of data
                        sys.stdout.buffer.flush()
  
            else:
                b1, b2 = self.split(b, boxes)
                boxes.append(b2)
                boxes.append(b1)
        self.datasink.flush()
            
    def split(self, b, boxes):
        if b.w > b.h:
            # Split vertically if width is bigger than height
            # Tried numpy random number generator, found to be twice as slow as the Python default generator
            split_size = b.w * uniform(self.split_range, 1 - self.split_range)
            b1 = BoxWithDepth(b.depth+1,b.x, b.y, split_size, b.h)
            b2 = BoxWithDepth(b.depth + 1, b.x + split_size, b.y, b.w - split_size, b.h)
        else:
            # Split horizontally if width is less than height
            split_size = b.h * uniform(self.split_range, 1 - self.split_range)
            b1 = BoxWithDepth(b.depth+1, b.x, b.y, b.w, split_size)
            b2 = BoxWithDepth(b.depth+1, b.x, b.y + split_size, b.w, b.h - split_size) 
        return b1, b2
    
    def dither_and_print(self, b):
        ditherx = b.w * uniform(0.0, self.dither)
        b.x += ditherx / 2
        b.w -= ditherx
        dithery = b.h * uniform(0.0, self.dither)
        b.y += dithery / 2
        b.h -= dithery

        self.datasink.writeBox([b.x, b.y], [b.x + b.w, b.y + b.h])
        
    def generate_point(self, i, prev_point):
        raise Exception("Cannot generate points with the ParcelGenerator")

def printUsage():
    sys.stderr.write(f"Usage: {sys.argv[0]} <key1=value1> ... \n")
    sys.stderr.write("The keys and values are listed below")
    sys.stderr.write("distribution: {uniform, diagonal, gaussian, parcel, bit, sierpinski}\n")
    sys.stderr.write("cardinality: Number of geometries to generate\n")
    sys.stderr.write("dimensions: Number of dimensions in generated geometries\n")
    sys.stderr.write("geometry: {point, box}\n")
    sys.stderr.write(" ** if geometry type is 'box' and the distribution is NOT 'parcel', you have to specify the maxsize property\n")
    sys.stderr.write("maxsize: maximum size along each dimension (before transformation), e.g., 0.2,0.2 (no spaces)\n")
    sys.stderr.write("percentage: (for diagonal distribution) the percentage of records that are perfectly on the diagonal\n")
    sys.stderr.write("buffer: (for diagonal distribution) the buffer around the diagonal that additional points can be in\n")
    sys.stderr.write("srange: (for parcel distribution) the split range [0.0, 1.0]\n")
    sys.stderr.write("dither: (for parcel distribution) the amound of noise added to each record as a perctange of its initial size [0.0, 1.0]\n")
    sys.stderr.write("affinematrix: (optional) values of the affine matrix separated by comma. Number of expected values is d*(d+1) where d is the number of dimensions\n")
    sys.stderr.write("compress: (optional) { bz2 }\n")
    sys.stderr.write("format: output format { csv, wkt, geojson }\n")
    sys.stderr.write("[affine matrix] (Optional) Affine matrix parameters to apply to all generated geometries\n")

class CommandLineArguments:
    def __init__(self, argv):
        self.argv = argv
    
    def getvalue(self, name):
        for arg in self.argv:
            parts = arg.split("=")
            if parts[0] == name:
                return parts[1]
        return None

def generate(distribution, cardinality, dimensions, geometryType,
            percentage=None, buffer=None, split_range=None,
            dither=None, probability=None, digits=None,
            affineMatrix=None, maxSize=None, seed=None
            ):
    generator = None
    if (distribution == "uniform"):
        generator = UniformGenerator(cardinality, dimensions)
    elif (distribution == "diagonal"):
        generator = DiagonalGenerator(cardinality, dimensions, percentage, buffer)
    elif (distribution == "gaussian"):
        generator = GaussianGenerator(cardinality, dimensions)
    elif (distribution == "parcel"):
        generator = ParcelGenerator(cardinality, dimensions, split_range, dither)
    elif (distribution == "bit"):
        generator = BitGenerator(cardinality, dimensions, probability, digits)
    elif (distribution == "sierpinski"):
        generator = SierpinskiGenerator(cardinality, dimensions)
    
    output_format = "wkt" #(form.getvalue("format") or "csv").lower()
    output = io.StringIO()
    
    if (output_format == "wkt"):
        datasink = WKTSink(output)
    elif (output_format == "csv"):
        datasink = CSVSink(output)
    elif (output_format == "geojson"):
        datasink = GeoJSONSink(output)
    else:
        raise Exception(f"Unsupported format '{output_format}'")

    if affineMatrix is not None:
        affineMatrix = [float(x) for x in affineMatrix.split(",") ]
    else:
        affineMatrix = None
    
    if seed is not None:
        rand.seed(seed)

    if maxSize is not None:
        maxSize=[float(x) for x in maxSize.split(",")]
    
    # Connect a point to box converter if the distribution only generated point but boxes are requested
    if (geometryType == 'box' and distribution != 'parcel'):
        datasink = PointToBoxSink(datasink, maxSize)
    # if (geometryType == 'polygon' and distribution != 'parcel'):
    #     polysize = float(form.getvalue("polysize"))
    #     maxseg = float(form.getvalue("maxseg"))
    #     datasink = PointToPolygonSink(datasink, maxseg, polysize)
    

    # If the number of parmaeters for the affineMatrix is correct, apply the affine transformation
    if (affineMatrix is not None and len(affineMatrix) == dimensions * (dimensions + 1)):
        datasink = AffineTransformSink(datasink, dimensions, affineMatrix)
    
    # Set the data sink (receiver) and run the generator
    generator.setSink(datasink)
    generator.generate()
    value = output.getvalue()
    output.close()
    return value

# def z_order(_x, _y):
#     # normalize to 0-1 then multiply by new maximum
#     size = 16
#     # x = int((_x - bounds[0]) / (bounds[2] - bounds[0]) * (2**16))
#     # y = int((_y - bounds[1]) / (bounds[3] - bounds[1]) * (2**16))
#     x = int(_x * (2**size))
#     y = int(_y * (2**size))
#     z = 0
#     # x = ctypes.c_uint32.from_buffer(ctypes.c_double(x)).value
#     # y = ctypes.c_uint32.from_buffer(ctypes.c_double(y)).value
#     for bit_position in range(size):
#         z |= (x & 1) << (2 * bit_position)
#         z |= (y & 1) << ((2 * bit_position) + 1)
#         x >>= 1
#         y >>= 1
#         bit_position += 1
#     return z

# def h_order(_points, p = 8, n = 2):
#     hilbert_curve = HilbertCurve(p, n) # side length of 2^p and n dimensions
#     points = np.zeros(_points.shape)
#     points[:, 0] = np.round(_points[:, 0] * ((2**p)-1))
#     points[:, 1] = np.round(_points[:, 1] * ((2**p)-1))
#     points = points.astype(np.int32).tolist()
#     return hilbert_curve.distances_from_points(points)


def get_geometry_features(geometry):
    return np.array(
        [
          geometry.centroid.x, # x-coord
          geometry.centroid.y, # y-coord
          0.0, # att
        ]
    )
def get_boundary(b1, b2):
    b = np.zeros(4)
    b[0] = min(b2[0], min(b1[0], b2[2]))
    b[1] = min(b2[1], min(b1[1], b2[3]))
    b[2] = max(b2[2], max(b1[0], b2[2]))
    b[3] = max(b2[3], max(b1[1], b2[3]))
    return b

def get_diagonal(b):
    p1 = b[[0,1]]
    p2 = b[[2,3]]
    return np.linalg.norm(p1 - p2)

def generate_gaussian_grid(x, y, var, dataset_points):
    points_file = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/gaussian_grids/%.2f_%.2f_%.2f_points.npy' % (x, y, var)
    values_file = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/gaussian_grids/%.2f_%.2f_%.2f_values.npy' % (x, y, var)
    if os.path.isfile(points_file):
        points = np.load(points_file)
        values = np.load(values_file)
    else:
        size = 1000
        mean = [x, y]
        cov = [[var, 0], [0, var]]  # Covariance matrix

        # Generate 2D Gaussian distributed data
        data = np.random.multivariate_normal(mean, cov, size)

        # Define the grid
        grid_size = size
        x = np.linspace(-0.05, 1.05, grid_size)
        y = np.linspace(-0.05, 1.05, grid_size)
        X, Y = np.meshgrid(x, y)

        # Compute the kernel density estimation (KDE) on the grid
        kde = stats.gaussian_kde(data.T)
        Z = kde(np.vstack([X.ravel(), Y.ravel()]))
        values = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        points = np.stack((X.ravel(),Y.ravel()), axis=1)
        np.save(values_file, values)
        np.save(points_file, points)
        # x = np.linspace(0, 1, 20000)
        # y = np.linspace(0, 1, 20000)
        # X, Y = np.meshgrid(x, y)
    return interpolate.griddata(points, values, dataset_points, method='cubic', rescale=False)
        # Reshape the density values to match the grid size
        # grid = Z.reshape(X.shape)

        # return grid

def box_counts2(features, distribution, cardinality, buffer, percentage, gaussianFeature):
    filename = "/rhome/msaee007/bigdata/pointnet_data/synthetic_data/data_points/%s_%d_%.2f_%.2f_%s_box_counts.npy" % (
    distribution, cardinality, buffer, percentage, gaussianFeature)
    if os.path.exists(filename):
        vals = np.load(filename)
        return vals[0], vals[1], vals[2]

    nodes_df = pd.DataFrame(features[:, [0, 1]], columns=["x", "y"])
    nodes_df.to_csv(f"/rhome/msaee007/bigdata/pointnet_data/synthetic_data/logs/{distribution}_points_tmp.csv", sep=",", index=False)
    FNULL = open(os.devnull, 'w')
    log_file = open(f'/rhome/msaee007/bigdata/pointnet_data/synthetic_data/logs/{distribution}.txt', 'w')
    output = str(subprocess.run([
                                    f"beast summary /rhome/msaee007/bigdata/pointnet_data/synthetic_data/logs/{distribution}_points_tmp.csv 'iformat:point(x,y)' -skipheader separator:, -boxcounting"],
                                shell=True, stdout=subprocess.PIPE, stderr=log_file).stdout)
    log_file.close()
                                # shell=True, stdout=subprocess.PIPE, stderr=FNULL).stdout)
    s1 = ""
    if 'e0' in output:
        start = output.index("\"e0\" :") + 6
        c = output[start]
        while c.isdigit() or c in ['.', ' ']:
            s1 += c
            start += 1
            c = output[start]
    s2 = ""
    if 'e2' in output:
        start = output.index("\"e2\" :") + 6
        c = output[start]
        while c.isdigit() or c in ['.', ' ']:
            s2 += c
            start += 1
            c = output[start]
    e0 = float(s1.strip()) if len(s1) else -999
    e2 = float(s2.strip()) if len(s1) else -999
    t1 = -1
    target = 'The operation summary finished in'
    with open(f'/rhome/msaee007/bigdata/pointnet_data/synthetic_data/logs/{distribution}.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            if target in l:
                t1 = float(l[l.find(target)+len(target)+1:l.rfind(' ')])

    np.save(filename, np.array([e0, e2, t1]))
    return e0, e2, t1

# def shrink_histogram(histogram):
#     l = histogram.shape[0]//2
#     return histogram.reshape((l, 2, l, 2)).sum(3).sum(1)
#     # length = histogram.shape[0]//2
#     # new_histogram = np.zeros((length,length))
#     # index_y = np.array(list(range(length))*length)
#     # index_x = []
#     # for i in range(length):
#     #     index_x += [i]*length
#     # index_x = np.array(index_x)
#     # new_histogram[index_x, index_y] += histogram[index_x*2, index_y*2]
#     # new_histogram[index_x, index_y] += histogram[index_x*2+1, index_y*2+1]
#     # new_histogram[index_x, index_y] += histogram[index_x * 2, index_y * 2 + 1]
#     # new_histogram[index_x, index_y] += histogram[index_x * 2 + 1, index_y * 2]
#     # return new_histogram

# def get_box_count_val(xs, ys, slope_variance, num_points, num_split_points=3, min_split_length=3):
#     split_points = np.array(slope_variance).argsort()[-num_split_points:].tolist() + [num_points]
#     split_points = sorted(split_points)
#     # print(slope_variance)
#     # print(slope_variance)
#     # print(len(slope_variance), num_points, split_points)
#     iPoint = 0
#     while iPoint < len(split_points):
#         split_length = split_points[iPoint] - split_points[iPoint - 1] if iPoint > 0 else split_points[iPoint]
#         if split_length < min_split_length:
#             del split_points[iPoint]
#         else:
#             iPoint += 1
#         # print(iPoint, split_points)
#     highest_support = 0
#     split_with_highest_support = -1
#     for iPoint in range(len(split_points) - 1):
#       support = split_points[iPoint + 1] - split_points[iPoint]
#       if  support > highest_support:
#         highest_support = support
#         split_with_highest_support = iPoint
#     # print(highest_support, iPoint)
#     sumx = 0.0
#     sumy = 0.0
#     sumxy = 0.0
#     sumx2 = 0.0
#     n = 0
#     _range = range(0, num_points) if split_with_highest_support == -1 else range(split_points[split_with_highest_support], split_points[split_with_highest_support + 1])
#     # print(_range)
#     for iPoint in _range:
#         x = xs[iPoint]
#         y = ys[iPoint]
#         # print(x, y)
#         n += 1
#         sumx += x
#         sumy += y
#         sumxy += x * y
#         sumx2 += x * x
#     # print(n, sumxy, sumx, sumy, sumx2, (n * sumxy - sumx * sumy), (n * sumx2 - sumx * sumx), '\n\n\n\n\n\n')
#     return (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx)


# def box_counts(histogram, bases=[0, 2]):
#     xs = []
#     ys = {b: [] for b in bases}
#     slope_variance = {b: [] for b in bases}
#     prev_slope = {0: None, 2: None}
#     grid_size = histogram.shape[0]
#     while grid_size >= 1:
#         # print(histogram.shape, histogram.max(), histogram.min(), (histogram[(histogram > 0)]**0).sum())
#         r = 1.0/grid_size
#         xs.append(np.log(r))
#         for b in bases:
#             ys[b].append(np.log((histogram[(histogram > 0)]**b).sum()))
#         if len(xs) > 1:
#             slope = {b: (ys[b][-1] - ys[b][-2]) / (xs[-1] - xs[-2]) for b in bases}
#             # print(xs[-2:], {b: ys[b][:-2]}, slope)
#             if len(xs) > 2:
#                 for b in bases:
#                     slope_variance[b].append(math.atan(abs(slope[b] - prev_slope[b])))
#             prev_slope = slope
#         grid_size /= 2
#         if grid_size >= 1:
#             histogram = shrink_histogram(histogram)
#     # print(xs)
#     # print(ys),
#     # print(slope)
#     # print(slope_variance)
#     return [get_box_count_val(xs, ys[b], slope_variance[b], len(xs)) for b in bases]

def generate_graph(
        path,
        distribution,
        cardinality,
        dimensions=2,
        geometryType="point",
        # minGaussianFeatureCenter="0.5,0.5,0.1",
        # maxGaussianFeatureCenter="0.5,0.5,0.1",
        gaussianFeature="0.25,0.25,0.75,0.75,0.1",
        percentage=.5,
        buffer=.5,
        split_range=None,
        dither=None,
        probability=None,
        digits=None,
        affineMatrix=None,
        maxSize=None,
        seed=None,
        # graph_size=10000, # maximum number of nodes in the output graph
        # alpha=0.1, # radius of KNN to build the graph
        # k=10, # nearest neighbors for KNN-Join for dataset output
        dis_min=0, # minimum for gaussian values
        dis_max=1, # maximum for gaussian values
        plot_data=True, # draw a plot of the geometries and the graph
    ):
    if dimensions != 2:
        raise Exception("Generator currently supports two dimensional geometries only.")
    
    labels = {
        "dataset_id": path,
        "tree_time": -1,
        "hotspots": {

        },
        "k_values": {

        },
        "box_counts": {

        },
    }
    
    ## The generate function uses spider to generate geoemtries in WKT format
    ## one geometry per line.

    if distribution == 'bit':
        buffer = digits
        percentage = probability
    elif distribution == 'parcel':
        buffer = split_range
        percentage = dither
    
    data_points_file = "/rhome/msaee007/bigdata/pointnet_data/synthetic_data/data_points/%s_%d_%.2f_%.2f_%s.npy" % (distribution, cardinality, buffer, percentage, gaussianFeature)
    if os.path.exists(data_points_file):
        features_matrix = np.load(data_points_file)
    else:
        t1=time.time()
        geometries_string = generate(distribution, cardinality, dimensions, geometryType,
                percentage, buffer, split_range,
                dither, probability, digits,
                affineMatrix, maxSize, seed)
        # print("[%.2f] Generated data." % (time.time()-t1))

        # Convert the newline delimited WKT to a list of Shapely geometries
        t1=time.time()
        geometries = np.empty((cardinality), dtype=object)
        features_matrix = np.zeros((cardinality, 3))
        # data_boundary = np.zeros(4)
        i = 0
        for line in geometries_string.split('\n'):
            wkt = line.strip()  # Remove leading/trailing whitespace or newline characters
            if len(wkt):
                if 'POLYGON' in wkt:
                    geometry = loads(wkt).centroid # we just want the centroids for parcel distribution
                else:
                    geometry = loads(wkt)
                features_matrix[i, :] = get_geometry_features(geometry)
                geometries[i] = geometry
                # if i == 0:
                #     data_boundary = np.array(geometry.bounds)
                # else:
                #     data_boundary = get_boundary(np.array(geometry.bounds), data_boundary)
                i+=1

        # diagonal = get_diagonal(data_boundary)
        # print("[%.2f] Parsed wkt and computed initial features." % (time.time()-t1))
        # print("Diagonal = %f, alpha = %f, r=%f" % (diagonal, alpha, diagonal*alpha))
        del geometries_string

        features_matrix[:, 0] = (features_matrix[:, 0] - features_matrix[:, 0].min()) / (features_matrix[:, 0].max()- features_matrix[:, 0].min())
        features_matrix[:, 1] = (features_matrix[:, 1] - features_matrix[:, 1].min()) / (features_matrix[:, 1].max()- features_matrix[:, 1].min())

        # Generate guassian feature based on grid-size and parameters
        t1=time.time()
        gaussianFeature = [float(x) for x in gaussianFeature.split(",")]
        shift_range = np.arange(-0.05, 0.05, 0.001)
        x_shift = np.random.choice(shift_range)
        y_shift = np.random.choice(shift_range)
        min_gaussian_vals = generate_gaussian_grid(gaussianFeature[0], gaussianFeature[1], gaussianFeature[4], features_matrix[:, :2]+(x_shift, y_shift))
        min_gaussian_vals = (min_gaussian_vals - min_gaussian_vals.min())/(min_gaussian_vals.max() - min_gaussian_vals.min())
        x_shift = np.random.choice(shift_range)
        y_shift = np.random.choice(shift_range)
        max_gaussian_vals = generate_gaussian_grid(gaussianFeature[2], gaussianFeature[3], gaussianFeature[4], features_matrix[:, :2]+(x_shift, y_shift))
        max_gaussian_vals = (max_gaussian_vals - max_gaussian_vals.min())/(max_gaussian_vals.max() - max_gaussian_vals.min())
        gaussian_vals = max_gaussian_vals-min_gaussian_vals
        # xs = np.clip(1000*(features_matrix[:, 0]), 0, 999).astype(int)
        # ys = np.clip(1000*(features_matrix[:, 1]), 0, 999).astype(int)
        features_matrix[:, 2] =  gaussian_vals #gaussian_grid[xs, ys]
        np.save(data_points_file, features_matrix)
    # print("[%.2f] Generated gaussian feature." % (time.time()-t1))
    features_matrix[:, 2] = (features_matrix[:, 2]-features_matrix[:, 2].min())/(features_matrix[:, 2].max()-features_matrix[:, 2].min()) * (dis_max - dis_min) + dis_min
    # return features_matrix
    e0, e2, t1 = box_counts2(features_matrix, distribution, cardinality, buffer, percentage, gaussianFeature)
    labels['box_counts'] = {'e0': e0, 'e2': e2, 'time': t1}
    # Compute avg gaussian value in value neighborhood (for dataset label only)
    # t1=time.time()
    # tree = KDTree(features_matrix[:, [0,1]], leaf_size=1)
    # indices = tree.query_radius(features_matrix[:, [1,2]], r=alpha*diagonal)
    # _, indices = tree.query(features_matrix[:, [0,1]], k=k)
    #
    # for i in range(features_matrix.shape[0]):
    #     avg_att = 0
    #     for j in range(0, min(k,len(indices[i]))):
    #         avg_att += features_matrix[indices[i][j], 3]
    #     avg_att /= min(k, len(indices[i]))
    #     features_matrix[i, 4] = avg_att
    # print("[%.2f] Computed avg(att) using k=%d" % ((time.time()-t1), k))


    #
    # Select random nodes
    # t1=time.time()
    # shuffled_indexes = np.arange(0, cardinality, 1)
    # np.random.shuffle(shuffled_indexes)
    # selected_nodes = features_matrix[shuffled_indexes[:graph_size], :]
    # selected_nodes = features_matrix
    # discarded_nodes = features_matrix[shuffled_indexes[graph_size:], :]
    # print("[%.2f] Selected random nodes." % (time.time()-t1))

    # # Create a KDTrees from geometry centroids
    t1=time.time()
    tree = KDTree(features_matrix[:, [0,1]], leaf_size=1)
    labels['tree_time'] = time.time()-t1
    # print("[%.2f] Built KDTree on selected nodes." % (time.time()-t1))
    #
    # # Get stats from discarded nodes
    # if discarded_nodes.shape[0] > 0:
    #     t1=time.time()
    #     indices = tree.query(discarded_nodes[:, [1,2]], k=1, return_distance=False)
    #     for i in range(discarded_nodes.shape[0]):
    #         neigh_id = indices[i][0]
    #         selected_nodes[neigh_id, 8] += 1
    #         selected_nodes[neigh_id, 9] += discarded_nodes[i, 7]
    #
    #     selected_nodes[:, 9] = np.where(selected_nodes[:, 8] != 0, selected_nodes[:, 9] / selected_nodes[:, 8], 0)
    #     print("[%.2f] Computed local_card and local_TA." % (time.time()-t1))
    #
    # # Perform range query
    # t1=time.time()
    # # (indices, distances) = tree.query_radius(selected_nodes[:, [1,2]], r=alpha*diagonal, return_distance=True)
    # (indices, distances) = tree.query(selected_nodes[:, [1,2]], k=k, return_distance=True)
    # print("[%.2f] Performed range query for graph edges." % (time.time()-t1))
    #
    #
    # # Convert the range query output to a sparse adjancency matrix
    # t1=time.time()
    # sparse_rows = []
    # sparse_cols = []
    # sparse_vals = []
    # for i in range(selected_nodes.shape[0]):
    #     for j in range(len(indices[i])):
    #         if indices[i][j] != i:
    #             sparse_rows.append(i)
    #             sparse_cols.append(indices[i][j])
    #             # sparse_vals.append(1-distances[i][j]/(alpha*diagonal))
    #             sparse_vals.append(1/distances[i][j])
    # print("[%.2f] Converted neigbhor lists to edge/weight lists." % (time.time()-t1))

    # t1=time.time()
    # for i in range(selected_nodes.shape[0]):
        # selected_nodes[i, 21] = z_order(selected_nodes[i, 0], selected_nodes[i, 1])
    # selected_nodes[:, 2] = h_order(selected_nodes[:, [0,1]])
    # min_value = np.min(selected_nodes[:, 2])
    # max_value = np.max(selected_nodes[:, 2])
    # selected_nodes[:, 2] = (selected_nodes[:, 2] - min_value) / (max_value - min_value)
    # selected_nodes = selected_nodes[selected_nodes[:, 2].argsort()]
    # selected_nodes[:, 0] = np.arange(0, graph_size)
    # print("[%.2f] Computed z-curve value, and sorted points." % (time.time()-t1))

    # Compute the k-Function values at different radii
    radii = [0.025, 0.05, 0.1, 0.25]
    n = features_matrix.shape[0]
    for r in radii:
        t1 = time.time()
        indices = tree.query_radius(features_matrix[:, [0, 1]], r=r, return_distance=False)
        value = sum([len(indices[i]) for i in range(n)]) / (n * (n - 1))
        t1 = time.time()-t1
        labels['k_values'][r] = {'value': value, 'time': t1}

    ks = [16, 32, 64]
    for k in ks:
        t1 = time.time()
        _, knn_index = tree.query(features_matrix[:, [0,1]], k=k)
        averages = np.zeros(knn_index.shape[0])
        for p in range(len(averages)):
            averages[p] = features_matrix[knn_index[p], 2].sum()/len(knn_index[p])
        arg_min = averages.argmin()
        arg_max = averages.argmax()
        t1 = time.time()-t1
        # print(gaussianFeature)
        # print(features_matrix[features_matrix[:,2].argmax()])
        # print(features_matrix[arg_max])
        # print(features_matrix[features_matrix[:,2].argmin()])
        # print(features_matrix[arg_min])
        labels["hotspots"][k] = {
            "min_val": averages[arg_min],
            "min_x": features_matrix[arg_min, 0],
            "min_y":features_matrix[arg_min, 1],
            "max_val": averages[arg_max],
            "max_x": features_matrix[arg_max, 0],
            "max_y": features_matrix[arg_max, 1],
            'time': t1
        }
    df= pd.DataFrame(features_matrix, columns=["x","y","att"])
    df.to_csv(path + ".csv", index=False)

    summary_path = path[:path.rfind('/')+1] + 'data_summary.csv'
    with open(summary_path, 'a') as file:
        file.write(json.dumps(labels) + '\n')
    


