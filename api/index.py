from flask import Flask, request#, Response
from api.utls import *

app = Flask(__name__)

@app.route( "/get-list", methods = ['POST'] )
def get_recommmendedMovies( ):
    return get_list(request=request)
	
@app.route( "get_auth" )
def get_token():
     return {'token':"eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4ODVhOGY3NGUxYjFhNjc0YTIwZDhmY2UzYzQ3OWJiOSIsInN1YiI6IjY1MmU3NTcxMDI0ZWM4MDExZTM1Njk5NSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.cY6LOPoZCTF5fuZ7h42N_bqHqau3XxgrHUI2o37Fvto"}

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == "__main__":
	app.run(debug=True)
    

