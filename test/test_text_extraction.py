from plate_analyzer.text_extraction import extract_minimums_from_text_box


class EmptyMinimumsBoxPlate:
    def get_text(self, option, clip):
        return "   "


def test_extract_minimums_from_empty_cell_returns_none():
    minimums = extract_minimums_from_text_box(
        box=None,
        minimum_type="S-ILS 17",
        plate=EmptyMinimumsBoxPlate(),
    )

    assert minimums is None
