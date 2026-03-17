"""CLI entrypoint for the cgft QA generation pipeline."""

import argparse

from cgft.qa_generation.cgft_pipeline import run_cgft_pipeline_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CgftPipeline from YAML config.")
    parser.add_argument("--config", required=True, help="Path to config file.")
    args = parser.parse_args()
    run_cgft_pipeline_from_config(args.config)


if __name__ == "__main__":
    main()
