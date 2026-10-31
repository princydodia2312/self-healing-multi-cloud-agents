"""MongoDB (semi-structured) source: schema inference, messy seed data, config bundle, credential
rules and the read_data_schema tool path. Hermetic — pymongo's client is mocked, no Atlas needed."""
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from bson import ObjectId

from agents import codegen
from utils import cloud_config as cc
from utils.config_utils import load_pipeline_bundle
from utils.mongo_schema import flattened_columns, infer_document_schema, to_jsonable

ROOT = Path(__file__).resolve().parent.parent

_DOCS = [
    {"_id": ObjectId(), "order_id": "ORD-1", "order_date": "2026-05-01T10:00:00",
     "customer": {"name": "Asha", "city": "Pune"},
     "items": [{"product": "Laptop", "price": 55000.0, "qty": 1}, {"product": "Mouse", "price": "500", "qty": 2}]},
    {"_id": ObjectId(), "order_id": "ORD-2", "customer": {"name": "Ravi"},          # no date, no city
     "items": [{"product": "Keyboard", "price": -10.0}]},                          # no qty
    {"_id": ObjectId(), "order_id": "ORD-3", "customer": {"name": "Meera", "city": None}},   # no items
]


class TestSchemaInference:
    def test_paths_types_and_presence(self):
        schema = {e["path"]: e for e in infer_document_schema(_DOCS)}
        assert schema["customer.city"]["present_pct"] == 67          # missing in 1 of 3 documents
        assert "sometimes missing" in schema["customer.city"]["note"]
        assert schema["items[].price"]["types"] == ["float", "string"]
        assert "MIXED TYPES" in schema["items[].price"]["note"]
        assert "ObjectId" in schema["_id"]["note"]
        assert "ARRAY" in schema["items"]["note"]

    def test_flattened_columns(self):
        cols = flattened_columns(infer_document_schema(_DOCS))
        assert {"order_id", "order_date", "customer_name", "customer_city", "product", "price", "qty"} <= set(cols)
        assert "items" not in cols and "customer" not in cols   # containers become leaf columns

    def test_to_jsonable(self):
        out = to_jsonable({"_id": ObjectId("6721a9f0c0ffee0000000001"),
                           "when": datetime.datetime(2026, 5, 1, 10, 0), "n": [1, {"x": None}]})
        assert out == {"_id": "6721a9f0c0ffee0000000001", "when": "2026-05-01T10:00:00", "n": [1, {"x": None}]}


class TestChaosDocuments:
    def test_messy_docs_hit_every_rule(self):
        from scripts.seed_chaos import create_mongo_chaos_docs

        docs = create_mongo_chaos_docs(500, seed=7)
        items = [it for d in docs for it in d.get("items", [])]
        assert len(docs) == 500
        assert any("order_id" not in d or not d["order_id"] for d in docs)          # completeness
        assert any("items" not in d for d in docs)                                   # no items list
        assert any(isinstance(it.get("price"), str) for it in items)                 # text prices
        assert any(isinstance(it.get("price"), (int, float)) and it["price"] <= 0 for it in items)
        assert any(it.get("qty") == 500 for it in items)                             # bulk outlier
        assert any(isinstance(it.get("qty"), int) and it["qty"] <= 0 for it in items)
        assert any(d.get("order_date") in ("2030-01-01T10:00:00", "not-a-date") for d in docs)
        assert any(not d["customer"].get("city") for d in docs)                      # city default
        # BSON-encodable: no numpy scalars leak into the documents
        import bson
        for d in docs[:50]:
            bson.encode(d)


class TestShopOrdersBundle:
    def test_bundle_loads(self):
        pipe, db, rules, infra = load_pipeline_bundle(
            str(ROOT), str(ROOT / "configs/pipelines/shop_orders_pipeline.yaml"))
        assert pipe["pipeline_id"] == "pipe_shop_orders_to_s3"
        assert db["db_type"] == "mongodb" and db["default_table"] == "raw_orders"
        assert len(rules["quality_standards"]) == 6
        assert {r["on_failure_action"] for r in rules["quality_standards"]} == {
            "DROP_RECORD", "EXCLUDE_AND_LOG", "DEFAULT_VALUE", "FLAG_AS_SUSPICIOUS"}
        # The shared pipelines IRSA role only grants arn:aws:s3:::*-insights-data
        assert pipe["aws_setup"]["bucket_name"].endswith("-insights-data")
        assert pipe["aws_setup"]["k8s_service_account_name"].endswith("-insights-sa")
        assert (ROOT / "configs/pipelines/shop_orders_objective.md").exists()

    def test_requirements_include_pymongo(self):
        lines = codegen.render_requirements("aws", "mongodb").splitlines()
        assert "pymongo" in lines and "psycopg2-binary" not in lines


class TestMongoCredentials:
    def test_env_fallback(self, monkeypatch):
        monkeypatch.setattr(cc, "_try_ssm", lambda c, k: None)
        monkeypatch.setattr(cc, "_load_bootstrap_outputs", lambda: {})
        monkeypatch.setenv("MONGO_URI", "mongodb+srv://u:p@x.mongodb.net/")
        monkeypatch.setenv("MONGO_DB_NAME", "shop_raw")
        assert cc.cloud_get("aws", "mongo_uri", db_type="mongodb").startswith("mongodb+srv://")
        assert cc.cloud_get("aws", "mongo_db_name", db_type="mongodb") == "shop_raw"

    def test_validator_rejects_getenv_mongo_uri(self, tmp_path):
        from agents.tools import validate_generated_code
        f = tmp_path / "pipe_x.py"
        f.write_text('import os\nuri = os.getenv("MONGO_URI")\n', encoding="utf-8")
        out = str(validate_generated_code.invoke({"filename": str(f)}))
        assert "POLICY VIOLATION" in out and 'cloud_get("aws", "mongo_uri", db_type="mongodb")' in out


class TestReadDataSchemaMongo:
    def test_infers_schema_and_never_leaks_uri(self, monkeypatch):
        from agents import tools

        uri = "mongodb+srv://user:secretpw@cluster0.abc.mongodb.net/"
        monkeypatch.setattr(tools, "cloud_get",
                            lambda cloud, key, db_type="postgres", **kw: {"mongo_uri": uri,
                                                                          "mongo_db_name": "shop_raw"}.get(key))
        client = MagicMock()
        client.__getitem__.return_value.__getitem__.return_value.find.return_value = list(_DOCS)
        with patch("pymongo.MongoClient", return_value=client):
            out = tools.read_data_schema.invoke({"table_name": "raw_orders", "db_type": "mongodb"})
        assert out["status"] == "success" and out["documents_sampled"] == 3
        assert "customer_city" in out["flattened_columns"]
        assert any("items[].price" in s and "MIXED TYPES" in s for s in out["schema"])
        assert isinstance(out["sample_documents"][0]["_id"], str)
        assert "secretpw" not in str(out)

    def test_connection_error_redacts_uri(self, monkeypatch):
        from agents import tools

        uri = "mongodb+srv://user:secretpw@cluster0.abc.mongodb.net/"
        monkeypatch.setattr(tools, "cloud_get", lambda cloud, key, db_type="postgres", **kw: uri if key == "mongo_uri" else None)
        with patch("pymongo.MongoClient", side_effect=Exception(f"cannot reach {uri}")):
            out = tools.read_data_schema.invoke({"table_name": "raw_orders", "db_type": "mongodb"})
        assert "Database Error on mongodb" in out and "secretpw" not in out


def test_mongodb_standard_fits_embedding_budget():
    from utils.embedding_budget import check_file
    _, ok = check_file(ROOT / "knowledge_base/engineering/mongodb_extraction_standard.md")
    assert ok
