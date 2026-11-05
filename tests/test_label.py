from pipeline.label import diabetes_codes, diabetes_label


def test_code_list_is_the_118_descendants_of_73211009():
    codes = diabetes_codes()
    assert len(codes) == 118
    assert {"73211009", "44054006", "46635009", "427089005"} <= codes
    assert "15777000" not in codes and "714628002" not in codes  # prediabetes is not diabetes


def test_label_on_fixtures(tables):
    y = diabetes_label(tables["patients"], tables["conditions"])
    # p1 type 2 DM, p3 CF-related DM; p2 prediabetes only, p4 diabetic kidney code only -> 0
    assert y.to_dict() == {"p1": 1, "p2": 0, "p3": 1, "p4": 0, "p5": 0}
