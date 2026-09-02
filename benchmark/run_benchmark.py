"""Run the benchmark: audio in, a ranked comparison out.

    python run_benchmark.py --audio recordings/ --truth labels/ --models whisper,seamless
    python run_benchmark.py --manifest dataset/manifest.json --out results/run-01

Models are loaded one at a time and replicated across every available GPU, so
peak memory is one model's worth however many are being compared. The notebook
in `notebooks/` calls `run()` directly with the same arguments.
"""
import argparse
import sys

import bridge
import dataset
import leaderboard
import scoring
import settings
import transcribe


def run(items, models=None, devices=None, language=None, terms_path=None,
        include_semantic=False, on_progress=None, model_factory=None, **window_kwargs):
    """Transcribe `items` with each model, then score everything at once.

    Scoring happens after all transcription rather than per model, because
    `summarize()` compares models against each other and needs them together.

    `model_factory` overrides how models are built -- pass
    `transcribe.dry_run_factory` to exercise the harness without weights.
    """
    models = list(models or settings.DEFAULT_MODELS)
    devices = list(devices or transcribe.resolve_devices())
    terms = bridge.ClinicalTerms(terms_path)

    runs = []
    for key in models:
        runs.append(transcribe.transcribe_batch(
            key, items, devices=devices, language=language,
            model_factory=model_factory, on_progress=on_progress, **window_kwargs))

    results, summary = scoring.score_all(runs, items, terms, include_semantic)
    summary["devices"] = transcribe.describe_devices(devices)
    summary["windowing"] = {
        "window_sec": window_kwargs.get("window_sec", settings.WINDOW_SEC),
        "overlap_sec": window_kwargs.get("overlap_sec", settings.OVERLAP_SEC),
    }
    return runs, results, summary


def load_items(args):
    """Whichever way the caller described the dataset."""
    if args.manifest:
        return dataset.from_json(args.manifest)
    if args.audio:
        return dataset.from_directory(args.audio, args.truth)
    raise SystemExit("give either --manifest or --audio")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_argument_group("dataset")
    source.add_argument("--manifest", help="JSON array of {asset_id, audio, reference}")
    source.add_argument("--audio", help="directory of recordings")
    source.add_argument("--truth", help="directory of <stem>.txt ground truth files")

    parser.add_argument("--models", default=None,
                        help="comma-separated stt registry keys (default: settings.DEFAULT_MODELS)")
    parser.add_argument("--devices", default=None,
                        help="comma-separated torch devices, e.g. cuda:0,cuda:1 (default: all GPUs)")
    parser.add_argument("--language", default=None, help="language code passed to each model")
    parser.add_argument("--out", default=str(settings.OUT_DIR), help="output directory")
    parser.add_argument("--terms", default=None, help="path to clinical_terms.json")
    parser.add_argument("--window-sec", type=float, default=settings.WINDOW_SEC)
    parser.add_argument("--overlap-sec", type=float, default=settings.OVERLAP_SEC)
    parser.add_argument("--semantic", action="store_true",
                        help="also compute BERTScore and semantic similarity (loads a model)")
    parser.add_argument("--list-models", action="store_true", help="print the registry and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="exercise decoding, windowing and scoring with a stub model, "
                             "so a wrong path costs seconds instead of a weights download")
    args = parser.parse_args(argv)

    if args.list_models:
        for key, spec in bridge.model_registry().items():
            print(f"  {key:24} {spec['model_id']}")
        return 0

    # Everything that can be checked cheaply is checked before any weights load:
    # a run that fails on the last line has already spent the GPU time.
    try:
        out_dir = leaderboard.check_writable(args.out)
    except OSError as error:
        raise SystemExit(str(error)) from None

    items = load_items(args)
    census = dataset.describe(items)
    if census["missing_audio"]:
        raise SystemExit(f"missing audio for: {', '.join(census['missing_audio'][:5])}")
    if not items:
        raise SystemExit("no recordings found")

    models = ([key.strip() for key in args.models.split(",") if key.strip()]
              if args.models else None)
    devices = transcribe.resolve_devices(args.devices)
    if args.dry_run:
        models = models or ["dry-run"]
        print("DRY RUN: stub model, no weights loaded -- scores are meaningless")

    print(f"{census['items']} recording(s): "
          f"{census['labelled']} labelled, {census['unlabelled']} unlabelled")
    for device in transcribe.describe_devices(devices):
        print(f"  {device['device']:9} {device['name']} ({device['total_vram_gb']} GB)")

    runs, results, summary = run(
        items, models=models, devices=devices, language=args.language,
        terms_path=args.terms, include_semantic=args.semantic,
        model_factory=transcribe.dry_run_factory if args.dry_run else None,
        on_progress=_progress, window_sec=args.window_sec, overlap_sec=args.overlap_sec)

    leaderboard.write(out_dir, results, summary, runs)

    print()
    print(leaderboard.to_markdown(summary))
    if summary["unscored_models"]:
        names = ", ".join(entry["model"] for entry in summary["unscored_models"])
        print(f"\nnot scored (no labelled recordings): {names}")
    print(f"\nwritten to {out_dir}")
    return 0


def _progress(transcript):
    state = transcript.error or f"{transcript.real_time_factor:.2f}x real time"
    print(f"  [{transcript.model}] {transcript.asset_id}: {state}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
