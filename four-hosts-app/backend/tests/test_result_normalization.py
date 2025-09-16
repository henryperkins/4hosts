import pytest

from services.result_adapter import ResultAdapter


def test_result_adapter_normalizes_object_and_dict():
    class Obj:
        def __init__(self):
            self.title = "Example Title"
            self.url = "https://example.com/page"
            self.snippet = "Snippet"
            self.content = "Body"
            self.source_api = "google_cse"
            self.credibility_score = 0.7
            self.domain = ""

    # Object form
    obj = Obj()
    a1 = ResultAdapter(obj)
    assert a1.has_required_fields() is True
    assert a1.url == "https://example.com/page"
    # Domain inferred from URL when missing
    assert a1.domain == "example.com"
    assert isinstance(a1.to_dict(), dict)

    # Dict form (missing domain should be inferred)
    d = {
        "title": "DTitle",
        "url": "https://sub.example.org/x",
        "snippet": "DSnippet",
        "content": "DContent",
        "source_api": "brave",
        "credibility_score": 0.6,
    }
    a2 = ResultAdapter(d)
    assert a2.has_required_fields() is True
    assert a2.domain == "sub.example.org"


def test_result_adapter_handles_missing_title():
    d = {"url": "https://host.tld/untitled"}
    a = ResultAdapter(d)
    # Falls back to last URL segment
    assert a.title.lower().find("untitled") >= 0

