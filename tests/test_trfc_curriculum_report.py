from src.trfc.curriculum_report import build_world_report_rows, summarize_world_report


def test_build_world_report_rows_uses_pipeline_estimates() -> None:
    cfg = {
        "io": {},
        "world": {
            "world_count": 2,
            "assignments": [
                {
                    "scene_json": "scene_000000.json",
                    "friction": {
                        "road_type": "AC",
                        "precip_type": "rain",
                        "precip_intensity_mmph": 1.0,
                        "water_film_mm": 0.05,
                    },
                },
                {
                    "scene_json": "scene_000001.json",
                    "friction": {
                        "road_type": "OGFC",
                        "precip_type": "rain",
                        "precip_intensity_mmph": 4.0,
                        "water_film_mm": 0.20,
                    },
                },
            ],
        },
        "ground": {
            "friction_values": [0.50],
            "friction_pipeline": {
                "enable": True,
                "defaults": {
                    "v_ref_mps": 13.89,
                    "s_ref_static": 0.15,
                    "s_ref_dynamic": 0.8,
                    "mu_eff_mode": "static",
                },
            },
        },
    }

    rows = build_world_report_rows(cfg)

    assert len(rows) == 2
    assert rows[0].friction_source == "pipeline_estimate"
    assert rows[0].scene_json_name == "scene_000000.json"
    assert rows[0].effective_friction < rows[1].effective_friction
    assert rows[1].road_type == "OGFC"

    summary = summarize_world_report(rows)
    assert summary["world_count"] == 2
    assert summary["road_type_counts"] == {"AC": 1, "OGFC": 1}


def test_build_world_report_rows_falls_back_to_legacy_friction_values() -> None:
    cfg = {
        "io": {
            "scene_jsons": ["scene_000010.json", "scene_000011.json"],
        },
        "world": {
            "world_count": 2,
        },
        "ground": {
            "friction_values": [0.40, 0.75],
            "friction_pipeline": {
                "enable": False,
            },
        },
    }

    rows = build_world_report_rows(cfg)

    assert [row.scene_json_name for row in rows] == ["scene_000010.json", "scene_000011.json"]
    assert [row.friction_source for row in rows] == [
        "legacy_friction_values",
        "legacy_friction_values",
    ]
    assert [row.effective_friction for row in rows] == [0.40, 0.75]
