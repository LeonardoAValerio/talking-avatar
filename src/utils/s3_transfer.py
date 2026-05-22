"""
S3 Transfer — Upload/download files from AWS S3.

Implementation notes for Claude Code:
- Used in production (AWS Batch) to fetch input (avatar image) and upload output (final MP4)
- Uses boto3 S3 client
- Input: s3://bucket/sources/avatar.jpg → /tmp/job/avatar.jpg
- Output: /tmp/job/final.mp4 → s3://bucket/outputs/<job_id>.mp4
- Bucket name from config (pipeline.yaml) or env var AWS_S3_BUCKET
"""

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Return (bucket, key) from an s3:// URI."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {s3_uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (missing bucket or key): {s3_uri!r}")
    return bucket, key


def _s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 not installed. Add it to requirements.txt.") from exc
    return boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def download(s3_uri: str, local_path: str | Path) -> Path:
    """Download a file from S3 to a local path.

    Args:
        s3_uri: Source S3 URI (e.g. s3://my-bucket/inputs/avatar.jpg).
        local_path: Destination file path on the local filesystem.

    Returns:
        Path to the downloaded local file.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    bucket, key = _parse_s3_uri(s3_uri)
    client = _s3_client()

    logger.info("S3 download: %s → %s", s3_uri, local_path)

    # Get object size for progress logging
    head = client.head_object(Bucket=bucket, Key=key)
    size_mb = head["ContentLength"] / 1024**2

    client.download_file(bucket, key, str(local_path))

    logger.info("S3 download complete: %.1f MB → %s", size_mb, local_path)
    return local_path


def upload(local_path: str | Path, s3_uri: str) -> str:
    """Upload a local file to S3.

    Args:
        local_path: Source file path on the local filesystem.
        s3_uri: Destination S3 URI (e.g. s3://my-bucket/outputs/job-123.mp4).

    Returns:
        The destination S3 URI.

    Raises:
        FileNotFoundError: If local_path does not exist.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File to upload not found: {local_path}")

    bucket, key = _parse_s3_uri(s3_uri)
    client = _s3_client()

    size_mb = local_path.stat().st_size / 1024**2
    logger.info("S3 upload: %s (%.1f MB) → %s", local_path, size_mb, s3_uri)

    client.upload_file(str(local_path), bucket, key)

    logger.info("S3 upload complete: %s", s3_uri)
    return s3_uri


def build_output_uri(bucket: str, job_id: str, filename: str = "final.mp4") -> str:
    """Build the standard output S3 URI for a job."""
    return f"s3://{bucket}/outputs/{job_id}/{filename}"
