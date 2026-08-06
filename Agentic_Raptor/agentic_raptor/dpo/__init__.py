"""Pre-SPICE DPO preference ranking (candidate pool → rank → select)."""

#: feature-vector dimensionality (spec embedding + graph features + 12 scalars)
from agentic_raptor.core.specifications import DesignSpecifications as _Spec
from agentic_raptor.core.types import DeviceType as _Dev
from agentic_raptor.dpo.preference_pairs import (
    OutcomeStore,
    PreferencePair,
    build_pairs,
    compare,
    split_pairs,
)
from agentic_raptor.dpo.ranker import DPOConfig, DPORanker
from agentic_raptor.dpo.schemas import (
    CandidateFeatures,
    DPOLeakageError,
    OutcomeRecord,
    build_candidate_features,
)
from agentic_raptor.dpo.selector import SelectionResult, select_candidate

FEATURE_DIM = len(_Spec.NUMERIC_FIELDS) + (len(tuple(_Dev)) + 5) + 13

__all__ = [
    "FEATURE_DIM",
    "CandidateFeatures",
    "DPOConfig",
    "DPOLeakageError",
    "DPORanker",
    "OutcomeRecord",
    "OutcomeStore",
    "PreferencePair",
    "SelectionResult",
    "build_candidate_features",
    "build_pairs",
    "compare",
    "select_candidate",
    "split_pairs",
]
