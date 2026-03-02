from src.trfc import encode_weather_context, weather_context_dim


def test_encode_weather_context_ac() -> None:
    vec = encode_weather_context(
        water_film_mm=0.5,
        road_type="AC",
    )

    assert len(vec) == weather_context_dim()
    assert vec == [0.5, 1.0, 0.0, 0.0]


def test_encode_weather_context_ogfc() -> None:
    vec = encode_weather_context(
        water_film_mm=1.0,
        road_type="ogfc",
    )

    assert vec == [1.0, 0.0, 0.0, 1.0]


def test_encode_weather_context_unknown_road_type() -> None:
    vec = encode_weather_context(
        water_film_mm=0.25,
        road_type="unknown",
    )

    assert vec == [0.25, 0.0, 0.0, 0.0]
