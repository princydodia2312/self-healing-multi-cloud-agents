"""patch_project_file must expose a typed {old, new} schema (Gemini sent strings for an untyped
`list`, so every self-heal patch crashed with "'str' object has no attribute 'get'")."""
import json

from agents.tools import patch_project_file


def test_gemini_schema_items_are_old_new_objects():
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    decl = convert_to_genai_function_declarations([patch_project_file])[0].function_declarations[0]
    items = decl.parameters.properties["replacements"].items
    assert items.type.name == "OBJECT"
    assert set(items.properties) == {"old", "new"}


def test_dict_and_json_string_items_both_apply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    f = tmp_path / "scripts" / "x.py"
    f.write_text("a = 1\nb = 2\n")
    r1 = patch_project_file.invoke({"filename": "scripts/x.py",
                                    "replacements": [{"old": "a = 1", "new": "a = 10"}]})
    r2 = patch_project_file.invoke({"filename": "scripts/x.py",
                                    "replacements": [json.dumps({"old": "b = 2", "new": "b = 20"})]})
    assert r1.startswith("PATCH APPLIED") and r2.startswith("PATCH APPLIED")
    assert f.read_text() == "a = 10\nb = 20\n"
