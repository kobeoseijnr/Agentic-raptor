"""Post-SAC design ranking (publication v2).

Distinct from the Bradley-Terry ranker inside `mb_sac.spec_sizing`, which
orders knob vectors during sizing. This package compares COMPLETE SIZED
DESIGNS and selects the one that receives authoritative ngspice verification.
"""

from agentic_raptor.ranking.post_sac import (PostSACDesign,
                                             RankerCheckpointMissing,
                                             RankerInputError,
                                             compare, hard_safety_tier,
                                             measured_preference,
                                             ranker_accuracy, record_pair,
                                             tri_state, trusted_pairs)
from agentic_raptor.ranking.model import (FEATURE_NAMES, PostSACRanker,
                                          features)
from agentic_raptor.ranking.surrogate import predict_post_sac
from agentic_raptor.ranking.types import (AuthoritativeSpiceOutcome,
                                          PredictionLeakage,
                                          SurrogatePrediction,
                                          assert_no_leakage,
                                          checkpoint_sha256,
                                          directory_sha256, netlist_hash_of,
                                          outcome_from_sizing,
                                          required_constraint_names,
                                          state_dict_sha256)

__all__ = ["PostSACDesign", "SurrogatePrediction",
           "AuthoritativeSpiceOutcome", "compare", "hard_safety_tier",
           "tri_state", "measured_preference", "record_pair",
           "trusted_pairs", "ranker_accuracy", "RankerInputError",
           "RankerCheckpointMissing", "PredictionLeakage",
           "assert_no_leakage", "outcome_from_sizing", "netlist_hash_of",
           "predict_post_sac", "checkpoint_sha256", "state_dict_sha256",
           "directory_sha256", "required_constraint_names",
           "PostSACRanker", "features", "FEATURE_NAMES"]
