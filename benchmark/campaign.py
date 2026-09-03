"""The `plan` / `status` / `work` / `combine` commands.

Kept out of `run_benchmark.py` so the one-shot comparison stays a short script
and the resumable campaign, which needs a plan file and a ledger, does not have
to pretend to be one.
"""
import json
import pathlib

import bridge
import dataset
import ledger as ledger_module
import plan as plan_module
import session
import settings
import tiers
import transcribe

PLAN_FILE = "plan.json"
PLAN_CSV = "plan.csv"


def load_items(args):
    """Whichever way the dataset described itself.

    `--labels` points at a dataset folder's labels.csv, which is how the
    Spin_BuAli_DataSet repo ships: one row per recording, label in a column.
    """
    if getattr(args, "labels", None):
        return dataset.from_csv(args.labels)
    if args.manifest:
        return dataset.from_json(args.manifest)
    if args.audio:
        return dataset.from_directory(args.audio, args.truth)
    raise SystemExit("give one of --labels, --manifest or --audio")


def load_plan(out_dir) -> list[dict]:
    path = pathlib.Path(out_dir) / PLAN_FILE
    if not path.is_file():
        raise SystemExit(f"no plan at {path} — run `plan` first")
    return json.loads(path.read_text(encoding="utf-8"))


# --- plan ------------------------------------------------------------------
def command_plan(args) -> int:
    """Decide every run, place each model in the hardware, and write it down."""
    usable, cards = tiers.usable_vram()
    if not cards:
        usable, cards = args.assume_vram, args.assume_cards
        print(f"no GPU visible — planning for {cards} x {usable:.1f} GB as asked")
    else:
        print(f"{cards} GPU(s), ~{usable:.1f} GB usable each "
              f"(~{usable * cards:.1f} GB if a model is sharded)")

    stt_models = ([m.strip() for m in args.stt.split(",") if m.strip()]
                  if args.stt else list(bridge.model_registry()))
    llm_models = ([m.strip() for m in args.llm.split(",") if m.strip()]
                  if args.llm else list(plan_module.LLM_PARAMS))
    cloud = [m.strip() for m in (args.cloud or "").split(",") if m.strip()]

    runs = plan_module.build(
        stt_models, llm_models, usable, cards,
        pipelines=tuple(p.strip() for p in args.pipelines.split(",") if p.strip()),
        preprocessing=tuple(p.strip() for p in args.preprocessing.split(",") if p.strip()),
        max_slots=args.max_slots, cloud=cloud)

    out_dir = pathlib.Path(args.out)
    ledger_module.write_json(out_dir / PLAN_FILE, runs)
    ledger_module.write_csv(out_dir / PLAN_CSV, plan_module.to_rows(runs))

    summary = plan_module.summarize(runs)
    print(f"\n{summary['runs']} runs planned — {summary['runnable']} runnable here, "
          f"{summary['deferred']} deferred")
    print("\nplacement, by model:")
    seen = set()
    for run in runs:
        key = (run["tier"], run["llm_model"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {run['tier']}  {run['llm_model']:28} {run['placement']}")
    print(f"\nwritten to {out_dir / PLAN_CSV}")
    print("\nTier A runs unquantized. Tier B is quantized only as far as it had to be.")
    print("Tier C is the same models at full precision — planned, not run here;")
    print("its point is to measure what the quantization in tier B cost.")
    return 0


# --- status ----------------------------------------------------------------
def command_status(args) -> int:
    runs = load_plan(args.out)
    store = ledger_module.Ledger(args.out)
    status = store.status(runs)

    print(f"{status['completed']} of {status['planned']} runs complete\n")
    print(f"  {'tier':6} {'done':>6} {'pending':>8} {'deferred':>9}")
    for tier, counts in status["by_tier"].items():
        print(f"  {tier:6} {counts['done']:>6} {counts['pending']:>8} {counts['deferred']:>9}")

    pending = store.pending(runs)
    if pending:
        print(f"\nnext up:")
        for run in pending[:5]:
            print(f"  {run['run_id']}  {run['tier']}  {run['preprocessing']:13} "
                  f"{run['pipeline']:11} stt={'+'.join(run['stt_models']) or '-':24} "
                  f"llm={run['llm_model']}")
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more")
    else:
        print("\nnothing pending — every runnable configuration has a result")
    return 0


# --- work ------------------------------------------------------------------
def command_work(args) -> int:
    """Execute pending runs until the plan is done or the budget stops it."""
    runs = load_plan(args.out)
    items = load_items(args)
    census = dataset.describe(items)
    if census["missing_audio"]:
        raise SystemExit(f"missing audio for: {', '.join(census['missing_audio'][:5])}")

    budget = ledger_module.Budget(minutes=args.max_minutes, max_runs=args.max_runs)
    devices = transcribe.resolve_devices(args.devices)

    print(f"{census['items']} recording(s): {census['labelled']} labelled")
    for device in transcribe.describe_devices(devices):
        print(f"  {device['device']:9} {device['name']} ({device['total_vram_gb']} GB)")
    if args.max_minutes:
        print(f"session budget: {args.max_minutes:.0f} min — it stops between runs, "
              f"never mid-run")
    print()

    def announce(run, outcome):
        detail = (f"{outcome['seconds']:.0f}s, {outcome.get('reports', 0)} reports"
                  if outcome["status"] == "ok" else outcome.get("reason", ""))
        print(f"  [{outcome['status']:7}] {run['run_id']}  {run['tier']}  "
              f"{run['preprocessing']:13} {'+'.join(run['stt_models']) or '-':22} {detail}",
              flush=True)

    report = session.work_through(
        runs, items, args.out, budget=budget, tier=args.tier, devices=devices,
        model_factory=transcribe.dry_run_factory if args.dry_run else None,
        on_run=announce, window_sec=args.window_sec, overlap_sec=args.overlap_sec)

    print(f"\n{len(report['performed'])} run(s) this session, "
          f"{report['remaining']} still pending")
    if report["stopped_because"]:
        print(f"stopped: {report['stopped_because']}")
    print(f"results in {pathlib.Path(args.out) / 'runs'} — safe to end the session")
    return 0


# --- combine ---------------------------------------------------------------
def command_combine(args) -> int:
    """Rebuild the leaderboard from whatever runs are on disk."""
    result = session.combine(args.out)
    print(f"combined {result['runs']} run(s) into {result['rows']} leaderboard row(s)")
    print(f"  {pathlib.Path(args.out) / 'leaderboard.csv'}")
    print(f"  {pathlib.Path(args.out) / 'all_reports.csv'}")
    return 0


COMMANDS = {"plan": command_plan, "status": command_status,
            "work": command_work, "combine": command_combine}


def add_arguments(parser):
    """The campaign flags, added to `run_benchmark.py`'s parser."""
    parser.add_argument("--tier", choices=["A", "B"], default=None,
                        help="limit a work session to one tier")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop starting runs after this long (Kaggle caps at 12h)")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--stt", default=None, help="comma-separated STT registry keys")
    parser.add_argument("--llm", default=None, help="comma-separated local LLM keys")
    parser.add_argument("--cloud", default=None,
                        help="comma-separated cloud models, e.g. gemini:gemini-2.5-pro")
    parser.add_argument("--pipelines", default=",".join(plan_module.PIPELINES))
    parser.add_argument("--preprocessing", default=",".join(plan_module.PREPROCESSING))
    parser.add_argument("--max-slots", type=int, default=1,
                        help="how many STT engines one run may combine")
    parser.add_argument("--assume-vram", type=float, default=14.8,
                        help="GB per card to plan for when no GPU is visible")
    parser.add_argument("--assume-cards", type=int, default=2)
    parser.add_argument("--devices", default=settings.DEVICES)
    parser.add_argument("--window-sec", type=float, default=settings.WINDOW_SEC)
    parser.add_argument("--overlap-sec", type=float, default=settings.OVERLAP_SEC)
    return parser
