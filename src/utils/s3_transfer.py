"""
S3 Transfer — Upload/download files from AWS S3.

Implementation notes for Claude Code:
- Used in production (AWS Batch) to fetch input (avatar image) and upload output (final MP4)
- Uses boto3 S3 client
- Input: s3://bucket/sources/avatar.jpg → /tmp/job/avatar.jpg
- Output: /tmp/job/final.mp4 → s3://bucket/outputs/<job_id>.mp4
- Bucket name from config (pipeline.yaml) or env var AWS_S3_BUCKET
"""
