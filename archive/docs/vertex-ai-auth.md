# Vertex AI Authentication: Portable Setup

## Problem

Currently using `gcloud auth application-default login`, which stores credentials at `~/.config/gcloud/application_default_credentials.json` — machine-specific, not portable to other servers.

## Solution: Service Account Key File

### 1. Create a Service Account (one-time, from any machine with gcloud or GCP Console)

```bash
# Create service account
gcloud iam service-accounts create docvqa-sa \
    --display-name="DocVQA Service Account"

# Grant Vertex AI User role (minimum needed for inference)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:docvqa-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Download JSON key file
gcloud iam service-accounts keys create docvqa-sa-key.json \
    --iam-account=docvqa-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

Or via **GCP Console**: IAM & Admin > Service Accounts > Create > Keys tab > Add Key > JSON.

### 2. Use on Any Server

Store the key file somewhere safe (outside the repo), then set one env var:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/docvqa-sa-key.json"
```

That's it. Zero code changes needed — the Google SDK (used by LiteLLM internally) checks `GOOGLE_APPLICATION_CREDENTIALS` first in its credential lookup chain.

## All Authentication Options

### Option A: Env var only (recommended — zero code changes)

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/docvqa-sa-key.json"
# Optionally also set these (LiteLLM reads them):
export VERTEXAI_PROJECT="YOUR_PROJECT_ID"
export VERTEXAI_LOCATION="global"
```

Your existing code works as-is. The Vertex SDK that LiteLLM/DSPy use will pick up the credentials automatically.

### Option B: LiteLLM `vertex_credentials` parameter

Pass credentials explicitly in LiteLLM calls. Useful if you want config-level control without env vars.

```python
import json

with open("/path/to/docvqa-sa-key.json") as f:
    creds = json.dumps(json.load(f))

response = litellm.completion(
    model="vertex_ai/gemini-3-flash-preview",
    messages=[...],
    vertex_credentials=creds,        # JSON string
    vertex_project="YOUR_PROJECT_ID",
    vertex_location="global",
)
```

To integrate with our codebase, you'd add `vertex_credentials` to `LMConfig` in `src/docvqa/types.py` and pass it through `to_dspy_lm()`.

### Option C: Workload Identity Federation (no key files, cloud-to-cloud)

If running on AWS/Azure/GKE, you can federate identity without a long-lived key file. More secure but requires cloud infrastructure setup. Pass the WIF credential config file via `vertex_credentials` or `GOOGLE_APPLICATION_CREDENTIALS`. Overkill for our use case.

## ADC (Application Default Credentials) Lookup Order

Google client libraries check credentials in this order:

1. `GOOGLE_APPLICATION_CREDENTIALS` env var (service account key or WIF config)
2. User credentials from `gcloud auth application-default login`
3. Attached service account (GCE, Cloud Run, GKE)
4. Metadata server (GCP infrastructure)

Setting `GOOGLE_APPLICATION_CREDENTIALS` overrides gcloud-based auth.

## Our Codebase Integration

Current auth flow: `configs/config.yaml` → `LMConfig` → `dspy.LM(**kwargs)` → LiteLLM → Vertex SDK → ADC.

Since DSPy/LiteLLM use the Vertex SDK which uses ADC, **Option A (env var) requires no code changes**. The `vertex_location: global` in our config is already passed through.

If we wanted Option B, the change would be:

| File | Change |
|------|--------|
| `src/docvqa/types.py` | Add `vertex_credentials: Optional[str] = None` to `LMConfig`, pass in `to_dspy_lm()` |
| `configs/config.yaml` | Add `vertex_credentials: ${oc.env:VERTEX_CREDENTIALS_PATH,null}` |

## Security Notes

- Never commit key files to git (add `*-key.json` to `.gitignore`)
- JSON keys don't expire by default (max 10 per service account)
- Grant minimum roles: `roles/aiplatform.user` for inference only
- Rotate keys periodically
- Google prefers Workload Identity Federation over key files when possible

## Quick Deploy Checklist

1. Copy `docvqa-sa-key.json` to the new server
2. `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/docvqa-sa-key.json"`
3. Run eval — done
