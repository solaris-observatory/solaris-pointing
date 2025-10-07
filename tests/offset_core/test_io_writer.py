
import types
from solaris_pointing.offset_core.io_writer import writer_factory

class FakeMetadata:
    def __init__(self, **kw): pass

def test_writer_includes_environment(monkeypatch, tmp_path):
    captured = {}
    def fake_write_offsets_tsv(output_path, md, rows, append):
        captured['output_path'] = output_path
        captured['append'] = append
        captured['rows'] = rows

    # Monkeypatch offset_io references inside the module
    import solaris_pointing.offset_core.io_writer as io_writer
    io_writer.Metadata = FakeMetadata
    io_writer.write_offsets_tsv = fake_write_offsets_tsv

    md = FakeMetadata()
    env = {"250106T010421": (-28.5, 690.2, 0.55)}
    ts_to_map = {"2025-01-06T01:04:21Z":"250106T010421"}

    writer = writer_factory(str(tmp_path/"offsets.tsv"), md, env, ts_to_map)
    writer("2025-01-06T01:04:21Z", 10.0, 20.0, 0.1, -0.2)

    assert captured['output_path'].endswith("offsets.tsv")
    assert captured['append'] is True
    assert len(captured['rows']) == 1
    row = captured['rows'][0]
    # Field names aligned to offset_io schema
    assert row.temperature_c == -28.5
    assert row.pressure_hpa == 690.2
    assert row.humidity_frac == 0.55
    assert row.timestamp_iso == "2025-01-06T01:04:21Z"
    assert row.azimuth_deg == 10.0
    assert row.elevation_deg == 20.0
    assert row.offset_az_deg == 0.1
    assert row.offset_el_deg == -0.2
