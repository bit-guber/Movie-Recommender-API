from flask import Flask, request

app = Flask(__name__)

# from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import json, time

file_path = "results/"
import os
print( os.listdir() )
max_display_movies = 10
movies = pd.read_csv( file_path + "movies.csv" ).set_index( "movieId" )
vectors = np.load(  file_path + "movie_vectors.npy" )
mapping_id = None
with open( file_path + "movie_id_to_vector_id.json", "r" ) as f:
    mapping_id = { int(k):int(v) for k,v in json.load( f ).items()}
top_movies = np.load( file_path + "top_movies.npy" )[:max_display_movies]

reverse_mapping_id = { v:k for k, v in mapping_id.items() }
each_movie_distances = dict()
def correlation( u, v, centered=False):
    if centered:
        umu = np.average(u)
        vmu = np.average(v)
        u = u - umu
        v = v - vmu
    uv = np.average(u * v)
    uu = np.average(np.square(u))
    vv = np.average(np.square(v))
    dist = 1.0 - uv / np.sqrt(uu * vv)
    # Return absolute value to avoid small negative value due to rounding
    return np.abs(dist)

def cosine( u, v ):
    return max(0, min(correlation(u, v, centered=False), 2.0))

def get_cosine_distance( target ):
    global each_movie_distances
    try:
        return each_movie_distances[target]
    except:
        target_vector = vectors[target]
        distances = np.zeros( vectors.shape[0], dtype = np.float32 )
        for i,x in enumerate(vectors):
            distances[i] = cosine( target_vector, x )
        each_movie_distances[target] = distances
        return each_movie_distances[target]

@app.route( "/get-list", methods = ['POST'] )
def get_list( ):
    viewed = request.get_json()["viewed"]
    if viewed is None :
        return top_movies.tolist()
    if len(viewed) == 0:
         return top_movies.tolist()
    cummalate = np.zeros( vectors.shape[0], dtype = np.float32 )
    viewed = [ mapping_id[x] for x in viewed ]
    start_time = time.time()
    for target in viewed:
        cummalate += get_cosine_distance( target  )
    print( time.time() - start_time )
    viewed = set(viewed)
    return [reverse_mapping_id[x] for x in np.argsort( cummalate ) if x not in viewed ][:max_display_movies]

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == "__main__":
	app.run(debug=True)
    

