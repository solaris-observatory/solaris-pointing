
import inspect
from solaris_pointing.offset_core.model import Site, MapInput, Config, WriterFn

def test_site_dataclass_defaults():
    s = Site(name="X", latitude_deg=0.0, longitude_deg=0.0)
    assert s.elevation_m == 0.0

def test_mapinput_dataclass_fields():
    m = MapInput(map_id="250106T010421", path_file="a.path", sky_file="a.sky",
                 map_timestamp_iso="2025-01-06T01:04:21Z")
    assert m.map_id.startswith("25")
    assert m.map_timestamp_iso.endswith("Z")

def test_config_defaults():
    c = Config()
    assert c.method in {"auto","gauss2d","boresight1d"}
    assert c.fwhm_deg > 0
    assert c.signal_min >= 0

def test_writerfn_signature():
    sig = inspect.signature(WriterFn)
    # Callable[[str, float, float, float, float], None]
    params = list(sig.parameters.values())
    assert len(params) == 5
