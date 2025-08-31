from models.synthesis_models import SynthesisContext as ModelSC
from services.answer_generator import SynthesisContext as ServiceSC


def test_synthesis_context_is_single_definition():
    # Ensure both modules expose the same class object
    assert ServiceSC is ModelSC

    # Basic construction works with required fields
    sc = ServiceSC(
        query="test",
        paradigm="analytical",
        search_results=[],
        context_engineering={},
    )
    assert sc.query == "test"
    assert sc.paradigm in {"analytical", "bernard", "revolutionary", "devotion", "strategic", "dolores", "teddy", "maeve"}

