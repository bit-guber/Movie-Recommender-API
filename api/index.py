from flask import Flask, request#, Response
# from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
# from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import json, time
from numpy.linalg import norm

file_path = "resources/"
import os
print( os.listdir() )
max_display_movies = 50
movies = pd.read_csv( file_path + "links.csv", dtype = { "tmdbId":"str", "movieId":np.int32 } ).set_index( "movieId" ).loc[:, "tmdbId" ].to_dict( ) # { k:movieId , v:tmdbId }
reverse_movies = { v: k for k,v in movies.items() } # { k:tmdbId, v:movieId }
vectors = np.load(  file_path + "movie_vectors.npy" )
mapping_id = None
with open( file_path + "movie_id_to_vector_id.json", "r" ) as f:
    mapping_id = { int(k):int(v) for k,v in json.load( f ).items()} # { k:movieId, v:fittedMovieId }
top_movies = np.load( file_path + "top_movies.npy" )[:max_display_movies].tolist() # top 100 movies of movieId

reverse_mapping_id = { v:k for k, v in mapping_id.items() } # { k:fittedMovieId, v:movieId }
each_movie_distances = dict()


def cosine_distance( u, v ):
    s = np.matmul(u,v.T)
    n = norm(u,keepdims=True, axis =1)*norm(v,keepdims=True, axis =1).T
    s = s / n
    s = ( s *-1) +1
    return s.sum(axis = 0)

def pre_processing( items ):
    return [reverse_movies[i] for i in items ]

def post_processing( items ):
    return [ movies[i] for i in items ]

@app.route( "/get-list", methods = ['POST'] )
def get_list( ):
    viewed = pre_processing(request.get_json()["viewed"])
    if viewed is None :
        return post_processing(top_movies)
    if len(viewed) == 0:
         return post_processing(top_movies)
    viewed_ids = [ mapping_id[x] for x in viewed ]
    viewed = np.array([ vectors[ x ] for x in viewed_ids ], dtype = np.float32)
    start_time = time.time()
    cummalate = cosine_distance( viewed ,vectors )
    print(cummalate.shape)
    print( time.time() - start_time )
    return post_processing([reverse_mapping_id[x] for x in np.argsort( cummalate ) if x not in viewed_ids ][:max_display_movies])

from flask import send_file
@app.route("/get_image")
def get_default_poster():
     return send_file("emptyPoster.jpg" )

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == "__main__":
	app.run(debug=True)
    

