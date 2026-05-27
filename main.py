from fastapi import FastAPI
from services.maps import get_route_data

app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "Traffic Backend Running"
    }


@app.get("/route")
def route(source: str, destination: str):

    return get_route_data(source, destination)