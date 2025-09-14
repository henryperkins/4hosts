from core.app import create_app
from utils.custom_docs import custom_openapi


def test_openapi_contains_evidence_bundle_schema():
    app = create_app()
    spec = custom_openapi(app)
    comps = spec.get("components", {}).get("schemas", {})
    assert "EvidenceBundle" in comps
    eb_schema = comps["EvidenceBundle"]
    assert "properties" in eb_schema
    props = eb_schema["properties"]
    assert "quotes" in props and "matches" in props and "focus_areas" in props

