# Deploying the Demo with Streamlit Community Cloud

[Streamlit Community Cloud](https://share.streamlit.io) hosts public GitHub repos for free — no credit card required.

## Steps

1. Make sure the repo is pushed to GitHub (branch `master`) and contains:
   - `streamlit_app.py` — the demo entry point;
   - `requirements.txt` — runtime dependencies (installs the local package with `-e .`);
   - `data/sample/` — sample data and demo models.

2. Open https://share.streamlit.io and sign in with your GitHub account.

3. Click **Create app** → **Deploy a public app from GitHub**.

4. Fill in:
   - Repository: `freshace/deepfmp-dg`
   - Branch: `master`
   - Main file path: `streamlit_app.py`

5. Click **Deploy**. The app URL will look like `https://deepfmp-dg.streamlit.app` (or `<your-account>-deepfmp-dg.streamlit.app`).

## Notes

- The first launch installs dependencies and downloads the SigLIP backbone (~1 GB); subsequent visits are fast because the model is cached.
- Free-tier memory is limited (~1 GB). The app is lightweight, but if you see out-of-memory errors, options are: retry, reduce uploaded image sizes, or move to a paid host.
- Every push to `master` can trigger a redeploy if you enable it in the app settings (or redeploy manually).
