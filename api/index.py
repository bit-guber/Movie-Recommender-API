from flask import Flask, request, Response

app = Flask(__name__)

# from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import json, time

file_path = "resources/"
import os
print( os.listdir() )
max_display_movies = 10
movies = pd.read_csv( file_path + "links.csv", dtype = { "tmdbId":"str", "movieId":np.int32 } ).set_index( "movieId" ).loc[:, "tmdbId" ].to_dict( ) # { k:movieId , v:tmdbId }
reverse_movies = { v: k for k,v in movies.items() } # { k:tmdbId, v:movieId }
vectors = np.load(  file_path + "movie_vectors.npy" )
mapping_id = None
with open( file_path + "movie_id_to_vector_id.json", "r" ) as f:
    mapping_id = { int(k):int(v) for k,v in json.load( f ).items()} # { k:movieId, v:fittedMovieId }
top_movies = np.load( file_path + "top_movies.npy" )[:max_display_movies].tolist() # top 100 movies of movieId

reverse_mapping_id = { v:k for k, v in mapping_id.items() } # { k:fittedMovieId, v:movieId }
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

def pre_processing( items ):
    return [reverse_movies[i] for i in items ]

def post_processing( items ):
    resp = Response( [ movies[i] for i in items ] )
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route( "/get-list", methods = ['POST'] )
def get_list( ):
    viewed = pre_processing(request.get_json()["viewed"])
    if viewed is None :
        return post_processing(top_movies)
    if len(viewed) == 0:
         return post_processing(top_movies)
    cummalate = np.zeros( vectors.shape[0], dtype = np.float32 )
    viewed = [ mapping_id[x] for x in viewed ]
    start_time = time.time()
    for target in viewed:
        cummalate += get_cosine_distance( target  )
    print( time.time() - start_time )
    viewed = set(viewed)
    return post_processing([reverse_mapping_id[x] for x in np.argsort( cummalate ) if x not in viewed ][:max_display_movies])

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == "__main__":
	app.run(debug=True)
    

