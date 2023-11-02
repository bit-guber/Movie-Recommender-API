# Movie recommendation Engine

This Project accomplish top 50 Movie suggest for user from previously liked movies with quick and efficiently as possible way.<br>
this project suggest from trained model embedding vectors for each movie that `live Application` [link](https://bit-guber-movie-recommender.vercel.app/). this only Backend API point made by [bit_guber](https://github.com/bit-guber/Portfolio) with Python + flask tools.

### It follows a few steps:

- [Load neccessary data](#setup)
- [Wait until new requests come in](#idle-state)
- [Delivering suggestion](#movie-proposition-producer)

## Setup

Install few vital python packages and run `api/index.py` file. this load all trained model embedding vectors and movies metadata to memory and also contains cosine distance with few metric for perform inference movie recommendation.

## Idle state

Everytime a user request movie List based previous liked movie List and passing preprocessing step then allow to recommendation system.

## Movie Proposition producer

Recommendation model feed previous liked movies and return a Movie List response sent back to users, Response contains high probability movies that similar to previous Liked Movie list.<br><br> Actually response produce by The famous SVD Machine learning algorithm based on Matrix Factorization and this same used [Simon Funk](https://sifter.org/~simon/journal/20061211.html) during the Netflix Recommended Engine Prize. also that produce efficient and more relate suggestion.
