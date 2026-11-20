#!/usr/bin/env python3
"""Teacher-side empirical MLP-input sampler for the unknown-order SwiGLU lane.

Records the real MLP input distribution (the residual-stream vectors entering
gate_proj/up_proj of a target layer) on a frozen 216-prompt corpus, per the
preregistered protocol in
docs/research/2026-11-19-unknown-order-empirical-swiglu-recovery.md.

This harness runs the model. It is teacher/witness-side only: the solver
never sees the corpus, the model name, the layer, or anything except the
recorded x values and the shape header.

The prompt corpus is a frozen table of 18 templates x 12 slot values (216
prompts, deterministic order). The template tables below are the tables the
preregistration freezes; changing them after a lane has run would be a
recorded deviation.

Usage (real model, Kaggle T4 / any CUDA host):
    python tools/llm_reader/empirical_activation_sampler.py \
        --model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --layer 15 \
        --output-dir <dir> --shuffle-seed 261121

Usage (offline synthetic manifold, local CPU sanity only, NOT a protocol input):
    python tools/llm_reader/empirical_activation_sampler.py \
        --offline-synthetic --dimension 4096 --rank 24 \
        --points 18000 --output-dir <dir> --shuffle-seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --- frozen prompt corpus tables (preregistration 2026-11-19) -----------------
# 18 templates x 12 slot values = 216 prompts, deterministic order.

TEMPLATES: list[tuple[str, list[str]]] = [
    (
        "The capital of {slot} has a population of about two million people. "
        "The old town lies on a river bank. Tourists visit the cathedral in "
        "summer. The university was founded centuries ago. Public transport "
        "connects the suburbs to the center. Local newspapers document the "
        "history of the region.",
        [
            "Portugal", "Norway", "Denmark", "Austria", "Hungary", "Bulgaria",
            "Croatia", "Slovakia", "Lithuania", "Latvia", "Estonia", "Slovenia",
        ],
    ),
    (
        "{slot} was invented in the twentieth century and changed daily life "
        "in ways nobody predicted. Early prototypes were expensive and rare. "
        "Mass production lowered the price within two decades. Competing "
        "designs disappeared quickly. Repair shops opened in every district. "
        "Schools taught the basic operation as a standard skill.",
        [
            "the microwave oven", "the ballpoint pen", "the transistor radio",
            "the compact disc", "the video cassette", "the digital camera",
            "the answering machine", "the fax machine", "the electric toothbrush",
            "the smoke detector", "the seat belt", "the contact lens",
        ],
    ),
    (
        "Summarize the following article about {slot} in three sentences. "
        "The article discusses recent trends and open controversies. Several "
        "researchers are quoted with opposing views. The historical background "
        "is given in the first section. Recent statistics close the argument. "
        "The author concludes with recommendations for practitioners.",
        [
            "urban cycling policy", "remote work productivity", "airline pricing",
            "battery recycling", "precision agriculture", "urban heat islands",
            "museum funding models", "coastal erosion", "speech recognition",
            "warehouse robotics", "plant-based food", "night shift scheduling",
        ],
    ),
    (
        "Translate the word '{slot}' into French and use it in a sentence. "
        "Afterwards list two synonyms and one antonym. Note any irregular "
        "plural forms. Mark the grammatical gender where it applies. Provide "
        "an example of formal usage and one of informal usage. Finish with a "
        "short note on common learner mistakes.",
        [
            "window", "harbour", "lantern", "meadow", "thunder", "cupboard",
            "willow", "cinnamon", "anvil", "ledger", "pebble", "wicker",
        ],
    ),
    (
        "What are the main economic activities in {slot} during the winter "
        "season? Describe the regional climate briefly. Explain how the "
        "workforce adapts to seasonal demand. Add one paragraph about "
        "transport logistics. Mention two regulatory constraints. Close with "
        "an outlook for the coming decade.",
        [
            "the alpine valleys", "the northern coast", "the inland plateau",
            "the river delta", "the border highlands", "the mining belt",
            "the lake district", "the steppe region", "the volcanic island",
            "the fjord settlements", "the desert oasis towns", "the taiga forest belt",
        ],
    ),
    (
        "Why does {slot} behave differently at low temperatures compared to "
        "room temperature? Explain the physical mechanism step by step. Give "
        "one measurable example with rough numbers. Describe what changes "
        "again near the boiling point. Note the practical engineering "
        "consequences. Finish with the standard textbook explanation.",
        [
            "rubber", "honey", "mercury", "silicone oil", "glass", "steel",
            "olive oil", "paraffin wax", "ethanol", "cast iron", "aluminium", "polycarbonate",
        ],
    ),
    (
        "def process(items):\n    return [item for item in items if item > {slot}]\n\n"
        "# Explain what this filter does. # Then rewrite it with an explicit\n"
        "# loop. # State the complexity. # Describe one edge case where the\n"
        "# two versions differ. # Add type hints to the final version.",
        [
            "0", "1", "2", "5", "10", "17", "23", "42", "99", "100", "255", "1000",
        ],
    ),
    (
        "SELECT name, total FROM orders WHERE year = {slot} GROUP BY name;\n\n"
        "-- What does this query compute? -- Rewrite it with a join against\n"
        "-- the customers table. -- Add a HAVING clause for totals above one\n"
        "-- thousand. -- Explain the execution order of the clauses.\n"
        "-- Name one index that would speed it up.",
        [
            "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018",
            "2019", "2020", "2021", "2022",
        ],
    ),
    (
        "User: I need help with {slot}.\n"
        "Assistant: I can help with that. Could you describe what you have "
        "already tried?\n"
        "User: I read the manual but the steps do not match the interface.\n"
        "Assistant: Let's start from the first screen and check each setting "
        "together.",
        [
            "printer setup", "router configuration", "keyboard shortcuts",
            "file backup", "screen calibration", "email filters",
            "font installation", "power settings", "sound recording",
            "font pairing", "disk cleanup", "clipboard history",
        ],
    ),
    (
        "Customer: My order {slot} has not arrived yet.\n"
        "Agent: Let me check the status. When did you place it?\n"
        "Customer: Nine days ago, standard delivery.\n"
        "Agent: The parcel left the depot and should arrive within two "
        "working days.",
        [
            "#10441", "#10442", "#10443", "#10444", "#10445", "#10446",
            "#10447", "#10448", "#10449", "#10450", "#10451", "#10452",
        ],
    ),
    (
        "List the steps required to {slot}, from preparation to completion. "
        "For each step state the tools needed and one common mistake. "
        "Estimate the time a beginner needs per step. Add a safety note for "
        "the risky steps. Describe how an expert shortens the same "
        "procedure. End with a checklist someone could print.",
        [
            "change a flat bicycle tire", "sharpen a kitchen knife",
            "repot a large houseplant", "sew a button on a coat",
            "paint a wooden fence", "unclog a sink drain",
            "assemble flat-pack shelves", "replace a door handle",
            "solder two wires together", "clean a camera lens",
            "hang a picture frame", "weatherproof a window",
        ],
    ),
    (
        "Write a shopping list for cooking {slot} for four people. Include "
        "fresh, dry and refrigerated goods in separate sections. Give rough "
        "quantities for each item. Add one optional substitute per main "
        "ingredient. Note which items keep for a week. Estimate the total "
        "budget at a mid-range store.",
        [
            "lentil soup", "roast chicken", "vegetable curry", "beef stew",
            "potato gratin", "fish soup", "mushroom risotto", "bean chili",
            "spinach pie", "pork goulash", "eggplant pasta", "chicken paella",
        ],
    ),
    (
        "{slot} je najveci grad u svom regionu i poznat je po istorijskim "
        "spomenicima. Stara varos lezi na reci. Turisti obilaze tvrdjavu "
        "leti. Univerzitet je osnovan pre vise vekova. Javni prevoz povezuje "
        "predgradja sa centrom. Lokalne novine prate istoriju kraja.",
        [
            "Beograd", "Novi Sad", "Nis", "Kragujevac", "Subotica", "Zrenjanin",
            "Pancevo", "Cacak", "Kraljevo", "Sabac", "Sombor", "Vranje",
        ],
    ),
    (
        "Die Stadt {slot} liegt im Sueden des Landes und ist fuer ihre "
        "Bruecken bekannt. Die Altstadt liegt am Fluss. Touristen besuchen "
        "den Dom im Sommer. Die Universitaet wurde vor Jahrhunderten "
        "gegruendet. Der Nahverkehr verbindet die Vororte mit dem Zentrum. "
        "Lokale Zeitungen dokumentieren die Geschichte der Region.",
        [
            "Freiburg", "Wuerzburg", "Bamberg", "Passau", "Heidelberg",
            "Tuebingen", "Konstanz", "Regensburg", "Augsburg", "Ulm",
            "Mainz", "Trier",
        ],
    ),
    (
        "Le quartier de {slot} se trouve pres du canal et est reconnu pour "
        "ses ateliers d artistes. Les rues sont etroites. Les cafes ouvrent "
        "tot le matin. Le marche du samedi attire la moitie de la ville. "
        "Les tramways desservent le quartier toutes les six minutes. Les "
        "archives locales conservent les plans anciens.",
        [
            "Belleville", "Batignolles", "Vaugirard", "Charonne", "Grenelle",
            "Passy", "Bercy", "Menilmontant", "Gobelins", "Saint-Marceau",
            "La Villette", "Plaisance",
        ],
    ),
    (
        "El barrio de {slot} se encuentra junto al rio y es conocido por sus "
        "talleres de artesanos. Las calles son estrechas. Los cafes abren "
        "temprano por la manana. El mercado del sabado atrae a media "
        "ciudad. Los tranvias llegan al barrio cada seis minutos. Los "
        "archivos locales conservan los planos antiguos.",
        [
            "Lavapies", "Malasana", "Chueca", "Arganzuela", "Chamartin",
            "Rios Rosas", "Trafalgar", "Almagro", "Goya", "Lista",
            "Recoletos", "Palomas",
        ],
    ),
    (
        "The {slot} sings quietly beneath the wooden lamp every second "
        "Thursday. Nobody remembers when the custom started. The chairs "
        "stand in rows facing the window. A small brass key sits on the "
        "shelf. The kettle whistles twice and stops. Outside the gravel "
        "counts itself in the dark.",
        [
            "teal lantern", "copper kettle", "velvet crow", "paper ox",
            "tin sparrow", "glass ferret", "oak sparrow", "brass heron",
            "ivory mole", "clay otter", "silk raven", "lead robin",
        ],
    ),
    (
        "Blorps the {slot} at midnight while the chairs applaud silently. "
        "The corridor nods in agreement. Three buttons discuss the weather. "
        "A spoon recites an old schedule. The carpet files a complaint about "
        "dust. The mirror refuses to comment on the incident. The staircase "
        "counts its own steps backwards.",
        [
            "mellow greln", "quiet dorvak", "soft mirlin", "pale vorrun",
            "large tellip", "small garrow", "round sorvin", "flat morvel",
            "deep tarlit", "thin querral", "wide bormen", "crisp lintov",
        ],
    ),
]


def build_corpus() -> list[str]:
    prompts: list[str] = []
    for template, slots in TEMPLATES:
        if len(slots) != 12:
            raise RuntimeError(f"template slot table must hold 12 values, got {len(slots)}")
        for slot in slots:
            prompts.append(template.format(slot=slot))
    if len(prompts) != 216:
        raise RuntimeError(f"corpus must hold 216 prompts, got {len(prompts)}")
    return prompts


def corpus_digest(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def split_indices(point_count: int, seed: int) -> dict[str, np.ndarray]:
    """Frozen 60/20/20 split of recorded point indices (deterministic)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(point_count)
    train_end = int(point_count * 0.6)
    selection_end = int(point_count * 0.8)
    return {
        "train": np.sort(order[:train_end]),
        "selection": np.sort(order[train_end:selection_end]),
        "validation": np.sort(order[selection_end:]),
    }


def record_real_model(
    model_name: str,
    layer: int,
    prompts: list[str],
    max_points: int,
) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    config = AutoConfig.from_pretrained(model_name)
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in {"mistral", "llama", "qwen2"}:
        raise RuntimeError(f"unsupported model_type {model_type!r} for empirical sampling")
    if getattr(config, "quantization_config", None):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    layers = model.model.layers
    if layer < 0 or layer >= len(layers):
        raise RuntimeError(f"layer {layer} outside range 0..{len(layers) - 1}")

    captured: list[torch.Tensor] = []

    def hook(_module, args):
        captured.append(args[0].detach().float().reshape(-1, args[0].shape[-1]).cpu())

    handle = layers[layer].mlp.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for prompt in prompts:
                text = prompt
                if tokenizer.chat_template is not None:
                    try:
                        text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    except Exception:
                        text = prompt
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                model(**inputs)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("no MLP inputs captured")
    points = torch.cat(captured, dim=0).numpy().astype(np.float32)
    meta: dict[str, object] = {
        "model": model_name,
        "modelType": model_type,
        "layer": layer,
        "hiddenSize": int(points.shape[1]),
        "layerCount": len(layers),
        "prompts": len(prompts),
    }
    return points, meta


def record_offline_synthetic(
    dimension: int,
    rank: int,
    point_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Local sanity source only; never a protocol input. A narrow random
    linear manifold plus modest background noise stands in for a real
    activation manifold so the downstream ladder machinery can be tested on
    CPU without any model download."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(0.0, 1.0, size=(rank, dimension)).astype(np.float64)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coords = rng.normal(0.0, 1.0, size=(point_count, rank))
    tail = rng.normal(0.0, 0.01, size=(point_count, dimension))
    points = (coords @ basis + tail).astype(np.float32)
    meta: dict[str, object] = {
        "model": "offline-synthetic-manifold",
        "layer": -1,
        "hiddenSize": dimension,
        "syntheticRank": rank,
        "syntheticSeed": seed,
    }
    return points, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical MLP-input sampler (teacher side)")
    parser.add_argument("--model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-points", type=int, default=32768)
    parser.add_argument("--shuffle-seed", type=int, default=261133)
    parser.add_argument("--offline-synthetic", action="store_true",
                        help="local sanity mode: synthetic manifold, no model")
    parser.add_argument("--dimension", type=int, default=4096,
                        help="offline synthetic mode only")
    parser.add_argument("--rank", type=int, default=24, help="offline synthetic mode only")
    parser.add_argument("--points", type=int, default=18000, help="offline synthetic mode only")
    parser.add_argument("--seed", type=int, default=11, help="offline synthetic mode only")
    args = parser.parse_args()

    prompts = build_corpus()
    if args.offline_synthetic:
        points, meta = record_offline_synthetic(args.dimension, args.rank, args.points, args.seed)
    else:
        points, meta = record_real_model(args.model, args.layer, prompts, args.max_points)

    if points.shape[0] > args.max_points:
        rng = np.random.default_rng(args.shuffle_seed)
        keep = rng.permutation(points.shape[0])[: args.max_points]
        points = points[np.sort(keep)]
    if not np.all(np.isfinite(points)):
        raise RuntimeError("recorded points contain non-finite values")

    splits = split_indices(points.shape[0], args.shuffle_seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "empirical-pool.npz"
    np.savez_compressed(
        pool_path,
        x=points,
        train=np.asarray(splits["train"], dtype=np.int64),
        selection=np.asarray(splits["selection"], dtype=np.int64),
        validation=np.asarray(splits["validation"], dtype=np.int64),
    )
    manifest = {
        "schemaVersion": 1,
        "kind": "gcl-empirical-mlp-input-pool",
        "corpus": {
            "templateCount": len(TEMPLATES),
            "slotsPerTemplate": 12,
            "promptCount": len(prompts),
            "sha256": corpus_digest(prompts),
            "prompts": prompts,
        },
        "pool": {
            "pointCount": int(points.shape[0]),
            "dimension": int(points.shape[1]),
            "maxPoints": int(args.max_points),
            "shuffleSeed": int(args.shuffle_seed),
            "splits": {name: int(len(idx)) for name, idx in splits.items()},
        },
        "model": meta,
        "offlineSynthetic": bool(args.offline_synthetic),
    }
    manifest_path = output_dir / "sampler-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    print(json.dumps({
        "poolPath": str(pool_path),
        "manifestPath": str(manifest_path),
        "pointCount": int(points.shape[0]),
        "dimension": int(points.shape[1]),
        "corpusSha256": manifest["corpus"]["sha256"],
        "poolSha256": pool_sha,
        "splits": manifest["pool"]["splits"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
