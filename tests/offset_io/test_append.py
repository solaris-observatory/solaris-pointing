from __future__ import annotations

from solaris_pointing.offset_io.tsv import write_offsets_tsv


def test_double_append_no_schema_error(
    tmp_path, md, sample_row, parse_noncomment_header_and_rows
):
    p = tmp_path / "out.tsv"
    # first write
    write_offsets_tsv(str(p), md, [sample_row], append=True)
    # second append with same schema (regression for the blank-line bug)
    write_offsets_tsv(str(p), md, [sample_row], append=True)

    text = p.read_text(encoding="utf-8")
    header, rows = parse_noncomment_header_and_rows(text)
    assert len(rows) == 2  # appended one more row
