#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging

import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact.
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading input artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)
    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])

    # Drop price outliers (accepted range is inclusive: min_price <= price <= max_price).
    # Missing values are intentionally not imputed here: the inference pipeline handles them.
    logger.info("Filtering prices outside the range [%s, %s]", args.min_price, args.max_price)
    idx = df["price"].between(args.min_price, args.max_price, inclusive="both")
    df = df[idx].copy()
    logger.info("Dropped %d price outliers, %d rows remain", (~idx).sum(), len(df))

    # Convert last_review from text to datetime
    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save the cleaned data and upload it to W&B
    logger.info("Saving cleaned data to clean_sample.csv")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Uploading artifact %s to W&B", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input W&B artifact with the raw data (e.g. sample.csv:latest)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output W&B artifact with the cleaned data (e.g. clean_sample.csv)",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact (e.g. clean_sample)",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum nightly price to keep (inclusive); cheaper listings are dropped as outliers",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum nightly price to keep (inclusive); more expensive listings are dropped as outliers",
        required=True,
    )

    args = parser.parse_args()
    go(args)
