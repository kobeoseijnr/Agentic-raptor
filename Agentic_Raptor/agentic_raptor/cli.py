"""RAPTOR command-line interface.

Run from the Agentic_Raptor directory (or after `pip install -e .`):

    python -m agentic_raptor.cli inspect-repository
    python -m agentic_raptor.cli parse-specification --config configs/experiments/smoke_test.yaml
    python -m agentic_raptor.cli validate --graph path/to/graph.json
    python -m agentic_raptor.cli smoke-test --config configs/experiments/smoke_test.yaml
    python -m agentic_raptor.cli run-episode --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_raptor.utils.logging import get_logger

logger = get_logger("agentic_raptor.cli")


def _cmd_check_dependencies(_args: argparse.Namespace) -> int:
    import importlib.util
    import os

    from agentic_raptor.spice.model_library import load_model_library
    from agentic_raptor.spice.ngspice_simulator import discover_ngspice, ngspice_version

    exe = discover_ngspice()
    print("== SPICE ==")
    if exe:
        print(f"  ngspice:        {exe}")
        print(f"  version:        {ngspice_version(exe) or 'unknown'}")
    else:
        print("  ngspice:        NOT FOUND (set spice.ngspice_exe or add ngspice_con/ngspice to PATH)")
    library = load_model_library(None, "generic_1u_level1")
    print(f"  model library:  built-in {library.label} (override via spice.model_library_path)")

    print("== Multimodal provider ==")
    for env_name in ("OPENAI_API_KEY", "AGENTIC_RAPTOR_LLM_API_KEY"):
        state = "set" if os.environ.get(env_name) else "NOT SET"
        print(f"  {env_name}: {state}")
    base = os.environ.get("AGENTIC_RAPTOR_LLM_BASE_URL")
    print(f"  AGENTIC_RAPTOR_LLM_BASE_URL: {base or '(default https://api.openai.com/v1)'}")

    print("== Optional packages ==")
    for module in ("torch", "networkx", "gymnasium", "yaml", "torch_geometric"):
        state = "available" if importlib.util.find_spec(module) else "missing"
        print(f"  {module}: {state}")
    return 0


def _cmd_build_netlist(args: argparse.Namespace) -> int:
    from agentic_raptor.core.circuit_graph import CircuitGraph
    from agentic_raptor.spice.model_library import load_model_library
    from agentic_raptor.spice.netlist_builder import build_circuit
    from agentic_raptor.utils.config import AgenticConfig

    config = AgenticConfig.from_yaml(args.config)
    graph = CircuitGraph.from_json(Path(args.graph).read_text(encoding="utf-8"))
    library = load_model_library(
        str(config.resolve_path(config.spice.model_library_path)) if config.spice.model_library_path else None,
        config.spice.technology_label,
    )
    built = build_circuit(graph, graph.sizing_state(), library, candidate_id="cli")
    print(built.circuit_text())
    return 0


def _cmd_run_spice(args: argparse.Namespace) -> int:
    from agentic_raptor.core.candidate import CircuitCandidate
    from agentic_raptor.core.circuit_graph import CircuitGraph
    from agentic_raptor.core.specifications import DesignSpecifications
    from agentic_raptor.core.types import GenerationSource
    from agentic_raptor.sizing.parameter_space import SizingParameterSpace
    from agentic_raptor.spice.ngspice_simulator import NgspiceSimulator
    from agentic_raptor.utils.config import AgenticConfig

    config = AgenticConfig.from_yaml(args.config)
    graph = CircuitGraph.from_json(Path(args.graph).read_text(encoding="utf-8"))
    spec_cfg = config.specification
    if spec_cfg.structured_specification_path:
        from agentic_raptor.specification import MultimodalDesignInput, parse_design_input

        spec, _report, _review = parse_design_input(
            MultimodalDesignInput(
                structured_specification_path=str(config.resolve_path(spec_cfg.structured_specification_path))
            ),
            defaults=dict(spec_cfg.defaults),
        )
    else:
        spec = DesignSpecifications(
            circuit_class="ota", technology="generic", supply_voltage=1.8, load_capacitance_f=1e-12
        )
        print("note: no structured spec in config; using benchmark defaults (ota/generic/1.8V/1pF)")
    candidate = CircuitCandidate.create(graph, spec, GenerationSource.MANUAL)
    if not candidate.sizing_state and not graph.sizing_state():
        space = SizingParameterSpace.from_graph(graph)
        candidate.sizing_state = space.denormalize(space.default_vector())
        print("note: graph carries no sizing; using parameter-space defaults")
    simulator = NgspiceSimulator(
        ngspice_exe=config.spice.ngspice_exe,
        technology_label=config.spice.technology_label,
        keep_workdirs=config.spice.keep_workdirs,
        seed=config.seed,
    )
    result = simulator.simulate(candidate, config.spice.analyses, config.spice.timeout_s)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 1


def _cmd_parse_schematic(args: argparse.Namespace) -> int:
    from agentic_raptor.specification.schematic import ProviderSchematicParser
    from agentic_raptor.topology_generation.provider import ProviderSettings, build_provider
    from agentic_raptor.utils.config import AgenticConfig

    config = AgenticConfig.from_yaml(args.config)
    settings = ProviderSettings(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_key_env=config.llm.api_key_env,
        base_url_env=config.llm.base_url_env,
        timeout_s=config.llm.provider_timeout_s,
        retries=config.llm.provider_retries,
    )
    missing = settings.missing_credential()
    if missing:
        print(f"cannot parse schematic: environment variable {missing} is not set")
        return 1
    parser = ProviderSchematicParser(build_provider(settings))
    result = parser.parse_rich(args.image)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_generate_topology(args: argparse.Namespace) -> int:
    from agentic_raptor.coordinator.coordinator import AgenticCoordinator
    from agentic_raptor.rag.retriever import seed_memory_for_smoke
    from agentic_raptor.specification import MultimodalDesignInput, parse_design_input
    from agentic_raptor.utils.config import AgenticConfig

    config = AgenticConfig.from_yaml(args.config)
    coordinator = AgenticCoordinator(config)
    s = config.specification
    design_input = MultimodalDesignInput(
        text=s.text,
        structured_specification_path=str(config.resolve_path(s.structured_specification_path)) if s.structured_specification_path else None,
        table_path=str(config.resolve_path(s.table_path)) if s.table_path else None,
        schematic_image_path=str(config.resolve_path(s.schematic_image_path)) if s.schematic_image_path else None,
        netlist_path=str(config.resolve_path(s.netlist_path)) if s.netlist_path else None,
    )
    spec, _report, _review = parse_design_input(design_input, defaults=dict(s.defaults))
    if len(coordinator.memory) == 0:
        seed_memory_for_smoke(coordinator.memory, spec)
    retrieval = coordinator.retriever.retrieve(spec)
    graphs = coordinator.generator.generate(
        spec, [e.entry for e in retrieval.entries], design_input, config.llm.number_of_candidates
    )
    out_dir = coordinator.output_dir
    for graph in graphs:
        validation = coordinator.validator.validate(graph)
        path = out_dir / f"generated_{graph.graph_id}.json"
        path.write_text(graph.to_json(), encoding="utf-8")
        print(f"{graph.graph_id}: valid={validation.is_valid} nodes={len(graph.nodes)} "
              f"confidence={graph.metadata.extra.get('confidence')} -> {path}")
        for issue in validation.issues:
            print(f"    [{issue.severity}] {issue.code}: {issue.message}")
    return 0


def _cmd_inspect_repository(_args: argparse.Namespace) -> int:
    from agentic_raptor.adapters.legacy_raptor import legacy_inventory, repo_root

    print(f"Legacy RAPTOR repo root: {repo_root()}")
    print("Legacy package availability (read-only imports):")
    for name, available in legacy_inventory().items():
        print(f"  {name:<14} {'available' if available else 'NOT FOUND'}")
    git_dir = repo_root() / ".git"
    print(f"Git repository: {'yes' if git_dir.is_dir() else 'NO (not under version control)'}")
    return 0


def _cmd_parse_specification(args: argparse.Namespace) -> int:
    from agentic_raptor.specification import MultimodalDesignInput, parse_design_input
    from agentic_raptor.utils.config import AgenticConfig

    config = AgenticConfig.from_yaml(args.config)
    s = config.specification
    design_input = MultimodalDesignInput(
        text=s.text,
        structured_specification_path=str(config.resolve_path(s.structured_specification_path)) if s.structured_specification_path else None,
        table_path=str(config.resolve_path(s.table_path)) if s.table_path else None,
        schematic_image_path=str(config.resolve_path(s.schematic_image_path)) if s.schematic_image_path else None,
        netlist_path=str(config.resolve_path(s.netlist_path)) if s.netlist_path else None,
    )
    spec, report, review = parse_design_input(design_input, defaults=dict(s.defaults))
    print("Modalities:", design_input.available_modalities())
    print("Canonical specification:")
    print(json.dumps(spec.to_dict(), indent=2))
    if report.conflicts:
        print(f"Conflicts detected ({len(report.conflicts)}):")
        for c in report.conflicts:
            print(f"  {c.field_name}: {c.values_by_source} → {c.resolved_value} ({c.resolved_source})")
    if review.checks:
        print("Plausibility review:")
        for check in review.checks:
            print(f"  [{check.severity}] {check.code}: {check.message}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from agentic_raptor.core.circuit_graph import CircuitGraph
    from agentic_raptor.topology_validation.validator import TopologyValidator

    graph = CircuitGraph.from_json(Path(args.graph).read_text(encoding="utf-8"))
    result = TopologyValidator().validate(graph)
    print(f"graph {graph.graph_id}: valid={result.is_valid} hash={result.graph_hash}")
    for issue in result.issues:
        print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    return 0 if result.is_valid else 1


def _cmd_run_episode(args: argparse.Namespace, smoke: bool = False) -> int:
    from agentic_raptor.coordinator.coordinator import AgenticCoordinator
    from agentic_raptor.utils.config import AgenticConfig

    config = AgenticConfig.from_yaml(args.config)
    coordinator = AgenticCoordinator(config)
    result = coordinator.run_episode()

    print("=" * 70)
    print(f"Episode {result.episode_id}: state={result.final_state} success={result.success}")
    print(f"Final reward: {result.final_reward}")
    if result.metrics:
        print("Nominal metrics:", json.dumps(result.metrics, indent=2, default=str))
    print(f"Decisions taken: {len(result.decisions)}")
    for d in result.decisions:
        print(f"  [{d['state']}] {d['decision']}: {d['reason']}")
    changed = result.update_report.get("parameters_changed", {})
    print("Learning verification (parameters changed):")
    for component, did_change in changed.items():
        print(f"  {component:<22} {'YES' if did_change else 'no'}")
    print(f"Run summary: {result.summary_path}")
    if smoke:
        required = ("policy_value_network", "sac_actor", "sac_critics", "dynamics_model")
        missing = [name for name in required if not changed.get(name)]
        if missing:
            print(f"SMOKE TEST FAILED: no parameter change in {missing}")
            return 1
        if result.error:
            print(f"SMOKE TEST FAILED: {result.error}")
            return 1
        print("SMOKE TEST PASSED: full pipeline executed with genuine learning updates.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_raptor", description="RAPTOR CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect-repository", help="show legacy RAPTOR module availability")
    sub.add_parser("check-dependencies", help="report ngspice, provider, and package availability")

    p_net = sub.add_parser("build-netlist", help="convert a circuit-graph JSON into SPICE cards")
    p_net.add_argument("--graph", required=True)
    p_net.add_argument("--config", required=True)

    p_rs = sub.add_parser("run-spice", help="simulate a circuit-graph JSON with real ngspice")
    p_rs.add_argument("--graph", required=True)
    p_rs.add_argument("--config", required=True)

    p_ps = sub.add_parser("parse-schematic", help="parse a schematic image via the multimodal provider")
    p_ps.add_argument("--image", required=True)
    p_ps.add_argument("--config", required=True)

    p_gt = sub.add_parser("generate-topology", help="generate topology candidates (mock or real LLM per config)")
    p_gt.add_argument("--config", required=True)

    p_s2 = sub.add_parser("run-stage2", help="run one Stage 2 episode (alias of run-episode with verification output)")
    p_s2.add_argument("--config", required=True)

    p_spec = sub.add_parser("parse-specification", help="parse multimodal design input from a config")
    p_spec.add_argument("--config", required=True)

    p_val = sub.add_parser("validate", help="validate a circuit-graph JSON file")
    p_val.add_argument("--graph", required=True)

    p_smoke = sub.add_parser("smoke-test", help="run the end-to-end mock pipeline with training verification")
    p_smoke.add_argument("--config", required=True)

    p_run = sub.add_parser("run-episode", help="run one agentic design episode")
    p_run.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "inspect-repository":
        return _cmd_inspect_repository(args)
    if args.command == "check-dependencies":
        return _cmd_check_dependencies(args)
    if args.command == "build-netlist":
        return _cmd_build_netlist(args)
    if args.command == "run-spice":
        return _cmd_run_spice(args)
    if args.command == "parse-schematic":
        return _cmd_parse_schematic(args)
    if args.command == "generate-topology":
        return _cmd_generate_topology(args)
    if args.command == "parse-specification":
        return _cmd_parse_specification(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "smoke-test":
        return _cmd_run_episode(args, smoke=True)
    if args.command in ("run-episode", "run-stage2"):
        return _cmd_run_episode(args, smoke=args.command == "run-stage2")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
