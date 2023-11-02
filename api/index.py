from flask import Flask, request#, Response
from api.utls import *

app = Flask(__name__)

@app.route( "/get-list", methods = ['POST'] )
def get_recommmendedMovies( ):
    return get_list(request=request)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
if __name__ == "__main__":
	app.run(debug=True)
    

