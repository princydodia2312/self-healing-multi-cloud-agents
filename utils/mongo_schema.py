"""Schema inference for MongoDB collections (semi-structured, schema-less JSON documents).

A SQL table has one fixed schema; a MongoDB collection does not — each document may nest
objects, hold arrays, omit fields, or store the same field as a number in one document and a
string in the next. `infer_document_schema` samples documents and reports every field path the
way the ETL script will see it after flattening:

    customer.city            nested object field   → column `customer_city`
    items[].price            field inside an array → one row per array element (`price`)

with the observed types and how often the field is present, so the Architect can write code that
survives missing fields and mixed types. `to_jsonable` makes sample documents printable/LLM-safe
(ObjectId / datetime / Decimal128 → str).
"""
from __future__ import annotations

import datetime
from collections import defaultdict


def _type_name(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (datetime.datetime, datetime.date)):
        return "date"
    if isinstance(v, dict):
        return "object"
    if isinstance(v, list):
        return "array"
    return type(v).__name__  # ObjectId, Decimal128, ...


def _walk(value, path: str, types, docs_with_path, seen_in_doc: set):
    if isinstance(value, dict):
        for k, v in value.items():
            child = f"{path}.{k}" if path else k
            _record(child, v, types, docs_with_path, seen_in_doc)
            _walk(v, child, types, docs_with_path, seen_in_doc)
    elif isinstance(value, list):
        for el in value:
            if isinstance(el, dict):
                for k, v in el.items():
                    child = f"{path}[].{k}"
                    _record(child, v, types, docs_with_path, seen_in_doc)
                    _walk(v, child, types, docs_with_path, seen_in_doc)


def _record(path, v, types, docs_with_path, seen_in_doc):
    types[path].add(_type_name(v))
    if path not in seen_in_doc:
        seen_in_doc.add(path)
        docs_with_path[path] += 1


def infer_document_schema(docs: list[dict]) -> list[dict]:
    """One entry per field path: {path, types, present_pct, note}. Sorted by path."""
    types: dict[str, set] = defaultdict(set)
    docs_with_path: dict[str, int] = defaultdict(int)
    for doc in docs:
        seen: set = set()
        _walk(doc, "", types, docs_with_path, seen)
    n = max(len(docs), 1)
    out = []
    for path in sorted(types):
        t = types[path]
        notes = []
        real = t - {"null"}
        if len(real) > 1:
            notes.append("MIXED TYPES — coerce (pd.to_numeric / pd.to_datetime with errors='coerce')")
        if "null" in t or docs_with_path[path] < n and "[]" not in path:
            notes.append("sometimes missing/null — never index it directly")
        if t == {"ObjectId"}:
            notes.append("BSON ObjectId — convert with .astype(str) before writing Parquet")
        if "array" in t:
            notes.append("ARRAY — explode/flatten (pd.json_normalize record_path) into one row per element")
        out.append({
            "path": path,
            "types": sorted(t),
            "present_pct": round(100 * docs_with_path[path] / n),
            "note": "; ".join(notes),
        })
    return out


def flattened_columns(schema: list[dict]) -> list[str]:
    """Column names after flattening: nested `a.b` → `a_b`; array element fields `items[].x` → `x`."""
    cols = []
    for entry in schema:
        if "object" in entry["types"] or "array" in entry["types"]:
            continue  # containers are flattened into their leaf fields
        path = entry["path"]
        col = path.split("[].")[-1] if "[]." in path else path
        cols.append(col.replace(".", "_"))
    return sorted(set(cols))


def to_jsonable(value):
    """Recursively convert BSON / datetime values to JSON-safe primitives (for samples & logs)."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)  # ObjectId, Decimal128, Binary, ...
