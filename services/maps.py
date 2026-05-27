import requests
import polyline


HEADERS = {
    "User-Agent": "TrafficPredictionSystem/1.0"
}


def get_coordinates(place_name):

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }

    response = requests.get(
        url,
        params=params,
        headers=HEADERS
    )

    if response.status_code != 200:
        raise Exception("Failed to fetch coordinates")

    data = response.json()

    if not data:
        raise Exception(f"Location not found: {place_name}")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])

    return lon, lat


def calculate_congestion(duration):

    if duration < 15:
        return "Low"

    elif duration < 30:
        return "Medium"

    else:
        return "High"


def get_route_data(source, destination):

    try:

        source_lon, source_lat = get_coordinates(source)
        dest_lon, dest_lat = get_coordinates(destination)

        url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{source_lon},{source_lat};{dest_lon},{dest_lat}"
        )

        params = {
            "overview": "full",
            "geometries": "polyline"
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception("Failed to fetch route")

        data = response.json()

        if "routes" not in data or len(data["routes"]) == 0:
            raise Exception("No route found")

        route = data["routes"][0]

        distance_km = route["distance"] / 1000
        duration_min = route["duration"] / 60

        encoded_polyline = route["geometry"]

        decoded_path = polyline.decode(encoded_polyline)

        congestion = calculate_congestion(duration_min)

        return {
            "source": source,
            "destination": destination,
            "distance": f"{distance_km:.2f} km",
            "estimated_time": f"{duration_min:.2f} mins",
            "congestion_level": congestion,
            "route_path": decoded_path
        }

    except Exception as e:

        return {
            "error": str(e)
        }