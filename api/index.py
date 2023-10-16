from flask import Flask, request

app = Flask(__name__)

from scipy.spatial.distance import cosine

import numpy as np
import pandas as pd
import json

file_path = "/results/"


max_display_movies = 10
movies = pd.read_csv( file_path + "movies.csv" ).set_index( "movieId" )
vectors = np.load(  file_path + "movie_vectors.npy" )
mapping_id = None
with open( file_path + "movie_id_to_vector_id.json", "r" ) as f:
    mapping_id = { int(k):int(v) for k,v in json.load( f ).items()}
top_movies = np.load( file_path + "top_movies.npy" )[:max_display_movies]

reverse_mapping_id = { v:k for k, v in mapping_id.items() }


def get_cosine_distance( target ):
    distances = []
    for x in vectors:
        distances.append( cosine( target, x ) )
    return np.array( distances )

@app.route( "/get-list", methods = ['POST'] )
def get_list( ):
    viewed = request.args.get( "viewed" )
    if len(viewed) == 0:
        return top_movies.tolist()
    cummalate = np.zeros( vectors.shape[0], dtype = np.float32 )
    viewed = [ mapping_id[x] for x in viewed ]
    for target in viewed:
        cummalate += get_cosine_distance( vectors[target]  )
    viewed = set(viewed)
    return [reverse_mapping_id[x] for x in np.argsort( cummalate ) if x not in viewed ][:max_display_movies]

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
	app.run(debug=True)
