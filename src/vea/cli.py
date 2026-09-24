"""Command line entry point.

    vea models          Preflight: which model weights are present or missing.
    vea config          Show the resolved settings for this environment.
    vea run <url>       Run stages 1-9 on one YouTube video.

``models`` and ``config`` deliberately avoid importing :mod:`vea.pipeline`, so
they work on a machine with no torch installed. That makes the preflight check
usable before committing to a full environment build.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vea import __version__
from vea.config import (
    MODEL_REGISTRY,
    STAGE_MODELS,
    configure_logging,
    extract_youtube_id,
    get_settings,
    missing_checkpoints,
    resolve_device,
    translation_model_ref,
)


def _cmd_config(_args: argparse.Namespace) -> int:
    settings = get_settings()
    print("Resolved settings (override with the VEA_* environment variables):")
    print(f"  VEA_DATA_DIR    {settings.data_dir}")
    print(f"  VEA_MODELS_DIR  {settings.models_dir}")
    print(f"  VEA_DEVICE      {settings.device}")
    print(f"  VEA_LOG_LEVEL   {settings.log_level}")
    try:
        ref = translation_model_ref(settings)
    except ValueError as exc:
        ref = f"INVALID - {exc}"
    print(f"  VEA_TRANSLATION_MODEL  {settings.translation_model} -> {ref}")

    try:
        import torch
    except ImportError:
        print("\ntorch is not installed; inference stages are unavailable.")
        return 0

    print(f"\ntorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  cuda:{i}  {name}  {total:.1f} GiB")
    return 0


def _cmd_models(_args: argparse.Namespace) -> int:
    settings = get_settings()
    missing = missing_checkpoints(settings)

    print(f"Model registry ({len(MODEL_REGISTRY)} entries)")
    print(f"Local checkpoints are looked up under {settings.models_dir}\n")

    for name, spec in sorted(MODEL_REGISTRY.items()):
        status = ("MISSING" if name in missing else "present") if spec.is_local else "hub"
        print(f"  [{status:>7}] {name}")
        print(f"            ref:     {spec.ref}")
        print(f"            purpose: {spec.purpose}")

    if not missing:
        print("\nAll declared local checkpoints are present.")
        return 0

    # Split the report, because the two cases call for different reactions: one
    # stops a run, the other is a registry entry kept only for its provenance.
    blocking = {n: s for n, s in missing.items() if n in STAGE_MODELS}
    unused = {n: s for n, s in missing.items() if n not in STAGE_MODELS}

    if blocking:
        print(f"\n{len(blocking)} checkpoint(s) missing that a stage needs:")
        for name, spec in blocking.items():
            print(f"\n  {name}")
            print(f"    stage:      {spec.purpose}")
            print(f"    provenance: {spec.provenance}")
        print("\nSee training/README.md to rebuild them.")

    if unused:
        print(
            f"\n{len(unused)} checkpoint(s) missing that no stage loads - these do not stop a run:"
        )
        for name, spec in unused.items():
            print(f"\n  {name}")
            print(f"    declared:   {spec.purpose}")
            print(f"    provenance: {spec.provenance}")

    # Non-zero only when a run would actually fail, so a setup script can gate
    # on this without tripping over a documented but unused entry.
    return 1 if blocking else 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    # required_only: a checkpoint no stage names cannot stop a run. Blocking on
    # one was a real defect - the refusal listed `emotion-ru-finetuned`, which
    # stage 7A does not load, in the same breath as telling the user 7A would
    # run fine without it.
    missing = missing_checkpoints(settings, required_only=True)
    if missing and not args.allow_missing_models:
        print("Refusing to start: checkpoints that a stage needs are missing.\n")
        for name, spec in missing.items():
            print(f"  {name} -> {spec.purpose}")
        print(
            "\nRun `vea models` for details, or pass --allow-missing-models to run "
            "the stages that do not need them."
        )
        return 1

    # Validate the URL before the pipeline is imported. The orchestrator loads
    # every model in an "INITIALIZING MODELS" block that runs *before* stage 1,
    # so a URL yt-dlp cannot parse currently costs a full NLLB-3.3B load - about
    # two minutes and 17 GB of downloads on a cold cache - before anyone finds
    # out. The stage already logs "Could not extract video ID" during setup and
    # then carries on regardless; this turns that warning into a refusal.
    #
    # The patterns live in vea.config, which imports nothing, so this check
    # costs nothing and runs in CI's fast job. vea.stages.download re-exports
    # the same function, so there is one pattern rather than two that drift.
    if not extract_youtube_id(args.url):
        print(f"Not a YouTube URL this pipeline can parse: {args.url!r}\n")
        print("Expected one of:")
        print("  https://www.youtube.com/watch?v=<11-character id>")
        print("  https://youtu.be/<11-character id>")
        print("  https://www.youtube.com/embed/<11-character id>")
        if "REAL_ID" in args.url or "<" in args.url:
            print("\nThat looks like a placeholder. Substitute a real video URL.")
        return 2

    # VEA_DEVICE=cuda on a machine torch sees no GPU on would otherwise surface
    # as a traceback from inside the orchestrator.
    try:
        resolve_device()
    except RuntimeError as exc:
        print(exc)
        return 2

    # Imported here, not at module level: this pulls in torch and transformers.
    from vea.pipeline import main as run_pipeline

    configure_logging(args.log_level)
    ok = run_pipeline(args.url)
    return 0 if ok else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The API needs the `api` extra:\n\n  uv sync --extra api\n", file=sys.stderr)
        return 2

    from vea.api.app import create_app

    configure_logging(None)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {args.host}. This API has no authentication and "
            "starts subprocesses that fetch URLs. Publish it to loopback only.",
            file=sys.stderr,
        )
    if args.reload:
        # Reload re-imports the module in a fresh process on every edit, so it
        # takes an import string; an app object built here would be discarded
        # and never serve a request. The environment variable is how the
        # downloads path reaches that re-import.
        os.environ["VEA_DOWNLOADS_DIR"] = str(Path(args.downloads))
        uvicorn.run("vea.api.app:app", host=args.host, port=args.port, reload=True)
        return 0

    uvicorn.run(
        create_app(downloads=Path(args.downloads)),
        host=args.host,
        port=args.port,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vea",
        description="Russian video emotion analysis pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"vea {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="show resolved settings and device info")
    p_config.set_defaults(func=_cmd_config)

    p_models = sub.add_parser("models", help="list models and flag missing checkpoints")
    p_models.set_defaults(func=_cmd_models)

    p_serve = sub.add_parser("serve", help="HTTP API for submitting and reading runs")
    p_serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address. Leave it on loopback: the API has no auth and hands "
        "URLs to yt-dlp. 0.0.0.0 is for inside a container, where it binds to "
        "the container's own namespace.",
    )
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--downloads", default="downloads", help="where runs are read from")
    p_serve.add_argument("--reload", action="store_true", help="auto-reload, for development")
    p_serve.set_defaults(func=_cmd_serve)

    p_run = sub.add_parser("run", help="run stages 1-9 on one YouTube URL")
    p_run.add_argument("url", help="YouTube video URL")
    p_run.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="start even though some checkpoints are missing; dependent stages will fail",
    )
    p_run.add_argument("--log-level", default=None, help="override VEA_LOG_LEVEL")
    p_run.set_defaults(func=_cmd_run)

    return parser


def _force_utf8_output() -> None:
    """Make stdout and stderr UTF-8 whatever the console's codepage is.

    The pipeline prints a stage arrow and, from stage 4 onward, Russian
    transcript text. On a Western-European Windows install ``sys.stdout`` is
    cp1252, which can encode neither, so the first one reached raises
    ``UnicodeEncodeError`` and kills a run that was otherwise healthy:

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

    Printing is not supposed to be a failure mode, so the streams are
    reconfigured rather than the messages sanitised - there are hundreds of the
    latter and they include arbitrary transcript text.

    ``errors="replace"`` rather than the default: a console that genuinely
    cannot render Cyrillic should show question marks, not abort stage 4.
    ``reconfigure`` is looked up rather than called directly because pytest's
    capture replaces the streams with objects that do not have it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
