"""
config/settings.py
------------------
Central configuration for the traffic data acquisition system.
All secrets come from environment variables / .env file.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


# ── Sub-configs ──────────────────────────────────────────────────────────────
class KafkaConfig(BaseModel):
    bootstrap_servers: List[str] = ["localhost:9092"]
    topics: dict = {
        "camera":  "traffic.camera.counts",
        "sensor":  "traffic.sensor.loop",
        "gps":     "traffic.gps.probes",
        "weather": "traffic.weather.current",
        "events":  "traffic.events.planned",
    }


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db:   int = 0
    ttl_seconds: int = 3600          # 1-hour feature cache


class InfluxConfig(BaseModel):
    url:    str = "http://localhost:8086"
    token:  str = ""
    org:    str = "traffic-org"
    bucket: str = "traffic-metrics"


class CameraConfig(BaseModel):
    frame_interval_sec: int = 60     # capture one frame per minute
    yolo_model:    str = "yolov8n.pt"
    confidence:    float = 0.40
    # Vehicle class IDs in COCO dataset
    vehicle_classes: dict = {
        2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
    }
    output_dir: Path = DATA_DIR / "camera"


class SensorConfig(BaseModel):
    poll_interval_sec: int = 30
    mqtt_broker:  str = "localhost"
    mqtt_port:    int = 1883
    mqtt_topic:   str = "traffic/sensors/#"
    output_dir: Path = DATA_DIR / "sensor"


class GPSConfig(BaseModel):
    poll_interval_sec: int = 300     # 5-minute travel-time windows
    output_dir: Path = DATA_DIR / "gps"


class WeatherConfig(BaseModel):
    poll_interval_sec: int = 3600    # hourly
    output_dir: Path = DATA_DIR / "weather"
    # OpenMeteo – no API key required
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    variables: List[str] = [
        "precipitation", "visibility", "wind_speed_10m",
        "temperature_2m", "weather_code"
    ]


class RoadNetworkConfig(BaseModel):
    output_dir: Path = DATA_DIR / "road_network"
    simplify: bool = True
    network_type: str = "drive"


# ── Master settings ──────────────────────────────────────────────────────────
class Settings(BaseSettings):
    # Location (default: Kolkata)
    city_name:  str   = "Kolkata, India"
    latitude:   float = 22.5726
    longitude:  float = 88.3639
    radius_m:   int   = 10_000        # 10 km radius

    # API keys (loaded from .env)
    here_api_key:        Optional[str] = Field(None, env="HERE_API_KEY")
    openweather_api_key: Optional[str] = Field(None, env="OPENWEATHER_API_KEY")
    kafka_enabled:       bool = False  # set True when Kafka is running

    # Sub-configs
    kafka:        KafkaConfig        = KafkaConfig()
    redis:        RedisConfig        = RedisConfig()
    influx:       InfluxConfig       = InfluxConfig()
    camera:       CameraConfig       = CameraConfig()
    sensor:       SensorConfig       = SensorConfig()
    gps:          GPSConfig          = GPSConfig()
    weather:      WeatherConfig      = WeatherConfig()
    road_network: RoadNetworkConfig  = RoadNetworkConfig()

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton
settings = Settings()

# Ensure output dirs exist
for cfg_attr in ["camera", "sensor", "gps", "weather", "road_network"]:
    getattr(settings, cfg_attr).output_dir.mkdir(parents=True, exist_ok=True)