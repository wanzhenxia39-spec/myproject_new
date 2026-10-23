"""Generate class-level A1 prototypes using the shared nine-attribute ontology."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from zshar.datasets import available_datasets, load_dataset_spec
from zshar.llm.transport import call_llm, resolve_api_key as resolve_transport_api_key
from zshar.paths import PROJECT_ROOT
from zshar.semantics.completion import (
    build_validation_report, dump_json, extract_json_object, validate_llm_output, write_text,
)
from zshar.semantics.ontology import DEFAULT_ONTOLOGY, load_ontology

EXTRA_ACTIVITIES = PROJECT_ROOT / "configs/activities"
DEFAULT_PROMPT_CONTEXT = PROJECT_ROOT / "configs/attributes/initial_vector_prompt_context.json"
PROMPT_POLICY = "semantic_a1_without_feature_groups_v2"
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_MODEL = "gpt-5.6-sol"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_api_key(api_format: str, base_url: str = "") -> str:
    """Resolve the key using the target endpoint's provider-specific priority."""
    return resolve_transport_api_key(api_format, base_url)


def available_initial_datasets() -> list[str]:
    return sorted(set(available_datasets()) | {p.stem for p in EXTRA_ACTIVITIES.glob("*.json")})


def render_readable_prompt(messages: list[dict]) -> str:
    """Render API messages for human review without JSON string escaping."""
    sections = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", ""))
        if message.get("role") == "user":
            try:
                content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
        sections.append(f"{'=' * 24} {role} {'=' * 24}\n\n{content.rstrip()}")
    return "\n\n".join(sections) + "\n"


def load_initial_context(dataset: str) -> dict:
    if dataset in available_datasets():
        spec = load_dataset_spec(dataset)
        return {"dataset": spec.key, "activities": spec.activity_names,
                "output_path": spec.attributes_path}
    path = EXTRA_ACTIVITIES / f"{dataset}.json"
    if not path.is_file():
        raise ValueError(f"Unknown initial-vector dataset: {dataset}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"dataset": dataset, "activities": payload["activities"],
            "output_path": PROJECT_ROOT / f"configs/attributes/activity_attribute_vectors_initial_{dataset}.json"}


def build_prompts(context: dict, *, ontology=None, prompt_context=None) -> list[dict]:
    ontology = ontology if ontology is not None else load_ontology()
    dataset = context["dataset"]
    activities = context["activities"]
    if not activities or len(activities) != len(set(activities)):
        raise ValueError("Activities must be nonempty and unique")
    attrs = ontology["attributes"]
    if len(attrs) != 9:
        raise ValueError("Initial A1 generation requires exactly nine attributes")
    if prompt_context is None:
        prompt_context = json.loads(DEFAULT_PROMPT_CONTEXT.read_text(encoding="utf-8"))
    dataset_prompt_context = prompt_context.get(dataset)
    if not isinstance(dataset_prompt_context, dict):
        raise ValueError(f"No initial-vector prompt context for dataset {dataset!r}")
    comparison_focus = dataset_prompt_context.get("comparison_focus")
    if not isinstance(comparison_focus, list) or not comparison_focus or any(
        not isinstance(item, str) or not item.strip() for item in comparison_focus
    ):
        raise ValueError(f"Invalid comparison_focus for dataset {dataset!r}")
    # A1 is a semantic baseline. Feature groups, concrete feature templates,
    # sensor inventories, and design-time bindings are intentionally excluded.
    # Executable attribute/group binding is decided only after A2 refinement.
    semantic_ontology = {"version": ontology["version"]}
    semantic_ontology["attributes"] = [
        {key: attr[key] for key in ("name", "definition", "levels")}
        for attr in attrs
    ]
    system = """You are an expert in biomechanics and wearable IMU activity recognition.
Generate one class-level initial semantic attribute vector A1 for every requested
activity. Follow this structured reasoning process internally:
1. Interpret each of the nine attributes strictly from its definition and ordinal levels.
2. Analyze the typical biomechanical motion pattern of every activity.
3. Compare the most easily confused activities in the complete activity list.
4. Assign all nine Low/Medium/High values using common anchors across activities.
5. Audit semantic consistency, exact keys, allowed values, and biomechanical support.

Think through those steps, but do not reveal private reasoning. Return only final JSON.
The supplied semantic ontology is the sole definition of A1. It supersedes the old
ten-attribute schema and must be used without adding, removing, renaming, merging,
splitting, or redefining attributes.

Rules for potentially confused attributes:
- motion_intensity measures motion amplitude and energy, not repetition rate or
  temporal consistency.
- motion_regularity evaluates consistency between repetitions and must not be
  inferred solely from cadence.
- smoothness concerns sustained temporal continuity over the window, whereas
  impulsiveness concerns localized short-duration peaks; they are not opposites.
- posture_inclination is average gravity-relative tilt, whereas
  posture_variability is the extent of its change within the window.
- rotational_motion and translational_motion can coexist and are not
  complementary labels.
- A short large displacement may increase motion_intensity or impulsiveness
  without implying High translational_motion.
- Repeated local oscillations or out-and-back movements with little sustained
  progression must not automatically be rated High translational_motion.

Semantic scope:
- Every value describes the activity as a whole and must remain invariant to
  sensor placement.
- Do not switch between whole-body posture and the orientation of the most
  active sensor across activities.
- posture_inclination describes the global load-bearing body configuration
  relative to gravity, rather than the orientation of an isolated moving limb.
- posture_variability describes changes in that global body configuration,
  rather than every local limb-orientation change.
- Local limb motion may contribute to motion_intensity and rotational_motion,
  but must not redefine the posture attributes.

The input contains only class-level semantic context, with no feature groups, feature
templates, attribute-to-feature mappings, sensor inventories, or measured feature values.
Do not infer or design any feature-group correspondence while assigning A1 values.
Do not turn body position, sensor type, or axis into an attribute or into a hidden
condition that changes an attribute's abstract physical meaning. Do not invent signal
measurements, thresholds, or unspecified mounting-axis interpretations.
Do not use train/test samples, subject information, or seen/unseen splits.
Describe typical activity semantics, not an individual recording. Distinguish similar
activities using the nine definitions. No two different activity types may have exactly
the same complete nine-attribute vector. If two draft vectors are identical, re-examine
the activities and assign at least one biomechanically justified difference; do not use
an arbitrary or sensor-placement-specific difference merely to satisfy this constraint.
Output all and only the exact requested activity keys and all nine attribute keys.
Each value must be exactly one of Low, Medium, High; no numeric values or extra fields.
"""
    user = json.dumps({
        "dataset": dataset,
        "activities": activities,
        "semantic_attribute_ontology": semantic_ontology,
        "dataset_specific_comparison_focus": comparison_focus,
        "activity_name_notes": dataset_prompt_context.get("activity_name_notes", {}),
        "comparison_instruction": (
            "Pay special attention to the dataset-specific comparison focus, and compare "
            "the most biomechanically confusable activities in this exact list "
            "before assigning levels. Ensure that every pair of different activities has "
            "a distinct complete vector, using only biomechanically justified differences "
            "under the nine supplied attribute definitions."
        ),
        "output_shape_example": {activities[0]: {a["name"]: "Low" for a in attrs}},
        "example_note": "This illustrates JSON shape only; assign actual levels independently.",
    }, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main(argv=None, *, default_dataset=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=available_initial_datasets(), default=default_dataset,
                        required=default_dataset is None)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output", type=Path, help="Defaults to the dataset profile's A1 JSON path")
    parser.add_argument("--prompt-out", type=Path)
    parser.add_argument("--prompt-text-out", type=Path,
                        help="Human-readable prompt; defaults beside --prompt-out as .txt")
    parser.add_argument("--dry-run", action="store_true", help="Save prompts without an API call")
    parser.add_argument("--response-json", type=Path,
                        help="Validate and save a previously obtained LLM response without an API call")
    parser.add_argument("--api-format", choices=["gemini", "openai"],
                        default=os.getenv("LLM_API_FORMAT", "openai"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", os.getenv("MODEL_NAME", DEFAULT_MODEL)))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming for OpenAI-compatible requests (default).",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.response_json:
        parser.error("Use either --dry-run or --response-json")
    if args.stream and args.api_format != "openai" and not (args.dry_run or args.response_json):
        parser.error("--stream is supported only with --api-format openai")
    context = load_initial_context(args.dataset)
    ontology = load_ontology(args.ontology)
    messages = build_prompts(context, ontology=ontology)
    output = args.output or context["output_path"]
    prompt_out = args.prompt_out or output.with_suffix(".prompt.json")
    prompt_text_out = args.prompt_text_out or prompt_out.with_suffix(".txt")
    dump_json(messages, str(prompt_out))
    write_text(render_readable_prompt(messages), str(prompt_text_out))
    if args.dry_run:
        print(
            f"{args.dataset}: saved API prompt to {prompt_out} and readable prompt to "
            f"{prompt_text_out}; no API call made"
        )
        return
    if args.response_json:
        raw = args.response_json.read_text(encoding="utf-8")
    else:
        api_key = resolve_api_key(args.api_format, args.base_url)
        raw = call_llm(messages=messages, api_format=args.api_format, api_key=api_key,
                       base_url=args.base_url, model=args.model, temperature=args.temperature,
                       timeout=args.timeout, ssl_no_verify=False, stream=args.stream)
    write_text(raw, str(output.with_suffix(".raw_response.txt")))
    names = [a["name"] for a in ontology["attributes"]]
    vectors = validate_llm_output(extract_json_object(raw), context["activities"], names)
    report = build_validation_report(vectors, context["activities"], names)
    report.update({"dataset": args.dataset, "attribute_set": "A1",
                   "ontology_version": ontology["version"], "ontology": ontology,
                   "prompt_path": str(prompt_out),
                   "readable_prompt_path": str(prompt_text_out),
                   "prompt_policy": PROMPT_POLICY,
                   "ontology_sha256": sha256(args.ontology),
                   "prompt_context_sha256": sha256(DEFAULT_PROMPT_CONTEXT),
                   "source": "response_file" if args.response_json else "llm",
                   "api_format": None if args.response_json else args.api_format,
                   "base_url": None if args.response_json else args.base_url,
                   "model": None if args.response_json else args.model,
                   "stream": None if args.response_json else args.stream})
    dump_json(vectors, str(output))
    dump_json(report, str(output.with_suffix(".report.json")))
    print(f"{args.dataset}: saved {len(vectors)} activities × {len(names)} attributes to {output}")


if __name__ == "__main__":
    main()
