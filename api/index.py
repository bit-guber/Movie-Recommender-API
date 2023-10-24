from flask import Flask, request, Response
# from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
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

from numpy import dot
from numpy.linalg import norm

def cosine( u, v ):
    s = np.matmul(u,v.T) #.reshape( 1, -1 )
    print(s.shape)
    n =norm(u,keepdims=True, axis =1)*norm(v,keepdims=True, axis =1).T
    print( n.shape )
    s = s / n
    print(s.shape)
    s = ( s *-1) +1
    print(s.shape)
    return s.sum(axis = 0)#.reshape(-1)
    # return (( dot(u,v)/(norm(u)*norm(v)) ) *-1) +1
def cosine_similarity(v1,v2):
    temp=  cosine( v1, v2) 
    # temp = np.zeros( len(vectors), dtype = np.float32 )
    # for i in range(v1.shape[0]):
        # temp+=  cosine( v1[i], v2) 

        # for j in range( v2.shape[0] ):
        #     temp[j]+=cosine( v1[i], v2[j] ) 
    return temp

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
    # cummalate = np.zeros( vectors.shape[0], dtype = np.float32 )
    viewed = [ mapping_id[x] for x in viewed ]
    start_time = time.time()
    # cummalate = cosine_distances( np.array([vectors[x] for x in viewed]) ,vectors )
    cummalate = cosine_similarity( np.array([vectors[x] for x in viewed]) ,vectors )
    print(cummalate.shape)
    print( time.time() - start_time )
    viewed = set(viewed)
    return post_processing([reverse_mapping_id[x] for x in np.argsort( cummalate ) if x not in viewed ][:max_display_movies])

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == "__main__":
	app.run(debug=True)
    

