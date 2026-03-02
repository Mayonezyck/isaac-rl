from src.trfc.friction_api import (
    AllWetRoadParameters,
    FrictionEstimate,
    FrictionInput,
    compute_mu_all_modified,
    estimate_friction,
)


def _estimate(*, road_type: str, water_film_mm: float) -> FrictionEstimate:
    return estimate_friction(
        FrictionInput(
            road_type=road_type,
            water_film_mm=water_film_mm,
            reference_speed_mps=13.89,
            slip_static=0.15,
            slip_dynamic=0.8,
        )
    )


def test_increasing_water_film_generally_reduces_friction() -> None:
    low_water = _estimate(road_type="AC", water_film_mm=0.2)
    high_water = _estimate(road_type="AC", water_film_mm=2.0)

    assert high_water.mu_static < low_water.mu_static
    assert high_water.mu_dynamic < low_water.mu_dynamic


def test_ogfc_exceeds_ac_for_same_water_film() -> None:
    ac = _estimate(road_type="AC", water_film_mm=0.5)
    ogfc = _estimate(road_type="OGFC", water_film_mm=0.5)

    assert ogfc.mu_static > ac.mu_static
    assert ogfc.mu_dynamic > ac.mu_dynamic


def test_estimate_friction_requires_explicit_water_film() -> None:
    try:
        estimate_friction(
            FrictionInput(
                precip_type="rain",
                precip_intensity_mmph=4.0,
                road_type="AC",
            )
        )
    except ValueError as exc:
        assert "water_film_mm must be provided" in str(exc)
    else:
        raise AssertionError("estimate_friction should require water_film_mm")


def test_contact_patch_area_is_explicit_and_affects_wet_contact_ratio() -> None:
    nominal = compute_mu_all_modified(
        v_ref=13.89,
        slip=0.15,
        h_w_mm=0.9,
        road_type="AC",
        params=AllWetRoadParameters(),
    )
    oversized_area = compute_mu_all_modified(
        v_ref=13.89,
        slip=0.15,
        h_w_mm=0.9,
        road_type="AC",
        params=AllWetRoadParameters(contact_patch_area_m2=4.0),
    )

    expected_nominal_area = (
        AllWetRoadParameters().contact_patch_length_m
        * AllWetRoadParameters().contact_patch_width_m
    )
    assert nominal.contact_patch_area_m2 == expected_nominal_area
    assert nominal.y_r < oversized_area.y_r
