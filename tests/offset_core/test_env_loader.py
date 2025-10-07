
from solaris_pointing.offset_core.env_loader import load_environment_csv, validate_environment
from solaris_pointing.offset_core.model import MapInput

def test_load_environment_csv_and_validate(tmp_path):
    csv = tmp_path / "environment.csv"
    csv.write_text("map_id,temperature_c,pressure_hpa,humidity_frac\n"
                   "250106T010421,-28.5,690.2,0.55\n"
                   "250106T020000,,689.1,\n"   # missing T and H → None
                   "ORPHAN,0,0,0\n")
    env = load_environment_csv(str(csv))
    assert env["250106T010421"] == (-28.5, 690.2, 0.55)
    assert env["250106T020000"] == (None, 689.1, None)

    maps = [
        MapInput("250106T010421","a.path","a.sky","2025-01-06T01:04:21Z"),
        MapInput("250106T020000","b.path","b.sky","2025-01-06T02:00:00Z"),
        MapInput("MISSING","c.path","c.sky","2025-01-06T03:00:00Z"),
    ]
    warnings = validate_environment(env, maps)
    # Should warn about ORPHAN and MISSING
    assert any("unknown map_id" in w.lower() for w in warnings)
    assert any("without environment" in w.lower() for w in warnings)
