"""
AWS Lambda Trigger — Receives API Gateway requests and starts Step Functions execution.

Implementation notes for Claude Code:
- Lambda validates the incoming JSON payload
- Starts a Step Functions execution with the payload as input
- Returns immediately (Lambda runs for <1 second)
- Lambda does NOT run GPU workloads

Expected payload:
{
    "text": "Texto a ser sintetizado...",
    "voice_id": "narrator_male_01",
    "avatar_s3_uri": "s3://bucket/sources/avatar.jpg",
    "compose_template": "lower_third",
    "compose_params": {
        "title": "Dr. João Silva",
        "subtitle": "Cardiologista"
    }
}

Environment variables:
    STATE_MACHINE_ARN: ARN of the Step Functions state machine
"""
