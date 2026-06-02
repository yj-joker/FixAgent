from services.manual_graph_extractor import select_schema_sections, normalize_graph_data


def test_select_schema_sections_keeps_fault_and_parts():
    sections = [
        {"section_title": "故障诊断与排除", "text_chunks": [{"text": "主轴异响时..."}], "tables": []},
        {"section_title": "前言", "text_chunks": [{"text": "本手册介绍..."}], "tables": []},
        {"section_title": "零件清单", "text_chunks": [], "tables": [{"rows": [["序号", "名称"]]}]},
    ]
    kept = [s["section_title"] for s in select_schema_sections(sections)]
    assert "故障诊断与排除" in kept
    assert "零件清单" in kept
    assert "前言" not in kept


def test_normalize_dedups_and_drops_empty():
    raw = {
        "components": [{"name": "主轴轴承"}, {"name": "主轴轴承"}, {"name": ""}],
        "faults": [{"name": "主轴异响", "relatedComponent": "主轴轴承"}],
        "solutions": [{"title": "更换轴承", "relatedFault": "主轴异响"}],
    }
    gd = normalize_graph_data(raw, manual_id=7, document_id="doc7", device_names=["数控车床"])
    assert gd["manualId"] == 7
    assert len(gd["components"]) == 1
    assert gd["faults"][0]["relatedComponentTempId"] == gd["components"][0]["tempId"]
    assert gd["solutions"][0]["relatedFaultTempId"] == gd["faults"][0]["tempId"]
