import argparse
import json
from pathlib import Path

DRAFTER = "deepseek-ai/dspark_qwen3_8b_block7"


def main() -> None:
    parser = argparse.ArgumentParser(prog="anemone-convert", description="Turn a Hugging Face LLM into an Anemone model folder")
    sub = parser.add_subparsers(dest="command", required=True)

    quant = sub.add_parser("quantize", help="GPTQ the target onto 4-bit lookup tables, one layer at a time (needs --extra export)")
    quant.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face id of the bf16 target")
    quant.add_argument("--chat-model", default="mlx-community/Qwen3-8B-4bit", help="MLX copy of the target that writes the chat calibration")
    quant.add_argument("--chat-calibration", type=Path, help="JSON of token ids to use instead of generating chat answers")
    quant.add_argument("--device", default="mps")
    quant.add_argument("--out", type=Path, required=True)

    target = sub.add_parser("target", help="export a palettized checkpoint as ANE chunk programs (needs --extra export)")
    target.add_argument("--checkpoint", type=Path, required=True, help="palettized checkpoint folder with palettization.json")
    target.add_argument("--context", type=int, default=1024, help="largest context rung; rungs double from 256")
    target.add_argument("--drafter", default=DRAFTER, help="drafter whose target_layer_ids set the taps")
    target.add_argument("--out", type=Path, required=True)

    drafter = sub.add_parser("drafter", help="export the DSpark drafter as a GPU program (needs --extra export)")
    drafter.add_argument("--drafter", default=DRAFTER)
    drafter.add_argument("--context", type=int, default=1024, help="target context the drafter must keep up with")
    drafter.add_argument("--out", type=Path, required=True, help="path of the .aimodel to write")

    pack = sub.add_parser("package", help="assemble a model folder from exported chunks and a drafter")
    pack.add_argument("--model", default="Qwen/Qwen3-8B", help="Hugging Face id of the target")
    pack.add_argument("--chunks", type=Path, required=True, help="folder with the exported chunk-*.aimodel target programs")
    pack.add_argument("--drafter", type=Path, required=True, help="the exported drafter .aimodel")
    pack.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "quantize":
        from anemone_convert.package import checkpoint
        from anemone_convert.quantize import chat_replies, documents, quantize

        source = checkpoint(args.model)
        saved = args.chat_calibration or args.out / "calibration-chat.json"
        if saved.exists():
            chat = json.loads(saved.read_text())
        else:
            chat = chat_replies(args.chat_model, source)
            args.out.mkdir(parents=True, exist_ok=True)
            saved.write_text(json.dumps(chat))
        quantize(source, documents(source) | chat, args.out, args.device)
        print(f"wrote {args.out}")
    elif args.command == "target":
        from huggingface_hub import snapshot_download

        from anemone_convert.target import export_target

        taps = json.loads((Path(snapshot_download(args.drafter, allow_patterns=["config.json"])) / "config.json").read_text())["target_layer_ids"]
        export_target(args.checkpoint, taps, args.context, args.out)
    elif args.command == "drafter":
        from anemone_convert.drafter import export_drafter

        export_drafter(args.drafter, args.context, args.out)
        print(f"wrote {args.out}")
    else:
        from anemone_convert.package import package

        package(args.model, args.chunks, args.drafter, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
