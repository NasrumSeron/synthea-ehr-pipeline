from pipeline.label import complication_codes, diabetes_codes, diabetes_label, mellitus_codes

COMPLICATIONS = {"127013003", "90781000119102", "157141000119108",
                 "368581000119106", "1551000119108", "97331000119101"}
NOT_DIABETES = {"15777000", "714628002", "80394007"}  # prediabetes x2, hyperglycaemia


def test_mellitus_list_is_the_118_descendants_of_73211009():
    codes = mellitus_codes()
    assert len(codes) == 118
    assert {"73211009", "44054006", "46635009", "427089005"} <= codes


def test_complication_list_is_the_6_observed_codes():
    assert complication_codes() == COMPLICATIONS
    assert not complication_codes() & mellitus_codes()


def test_label_codes_are_the_union_and_exclude_non_diabetes():
    codes = diabetes_codes()
    assert codes == mellitus_codes() | COMPLICATIONS
    assert len(codes) == 124
    assert not codes & NOT_DIABETES


def test_label_on_fixtures(tables):
    y = diabetes_label(tables["patients"], tables["conditions"])
    # p1 type 2 DM, p3 CF-related DM, p4 diabetic kidney code only -> 1; p2 prediabetes only -> 0
    assert y.to_dict() == {"p1": 1, "p2": 0, "p3": 1, "p4": 1, "p5": 0}
