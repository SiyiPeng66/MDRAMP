import pytest

from src.data.parse_public import parse_camp_csv


def test_camp_whitespace_headers_preserve_source_id(tmp_path):
    source = tmp_path / 'camp.csv'
    source.write_text('  Camp_ID,Seqence,Title\nCAM123,ACDEFGHIKL,example\n')
    assert parse_camp_csv(source)[0]['source_record_id'] == 'CAM123'


def test_camp_missing_source_id_column_is_not_silently_accepted(tmp_path):
    source = tmp_path / 'camp.csv'
    source.write_text('Sequence\nACDEFGHIKL\n')
    with pytest.raises(ValueError, match='Camp_ID'):
        parse_camp_csv(source)
