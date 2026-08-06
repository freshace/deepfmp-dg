# Deploying the Interactive Demo to Hugging Face Spaces

The Gradio demo can run on [Hugging Face Spaces](https://huggingface.co/spaces). Note: hosting Gradio Spaces now requires a Hugging Face PRO subscription (free accounts can only host static Spaces) ? see `docs/streamlit-deploy.md` for the free alternative.

## One-time setup

1. Build the Space bundle:
   ```bash
   python scripts/build_hf_space.py
   ```
   This creates `dist_space/` with `app.py`, `requirements.txt`, the package source, and the sample data/models.

2. Log in to Hugging Face (need an account + access token with `write` permission):
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login
   ```

3. Create the Space (name it e.g. `deepfmp-dg-demo`):
   ```bash
   huggingface-cli repo create deepfmp-dg-demo --type space --sdk gradio
   ```

4. Push the bundle:
   ```bash
   cd dist_space
   git init
   git add -A
   git commit -m "init"
   git remote add origin https://huggingface.co/spaces/<your-username>/deepfmp-dg-demo
   git push -u origin main
   ```

5. Open `https://huggingface.co/spaces/<your-username>/deepfmp-dg-demo` — the first build installs dependencies and downloads the SigLIP backbone (about 1 GB, cached afterwards).

## Keeping it updated

Rebuild and re-push whenever the repo changes:
```bash
python scripts/build_hf_space.py
cd dist_space
git add -A && git commit -m "sync"
git push
```

## Notes

- The Space card lives in `dist_space/README.md` (generated with the current Gradio version).
- The demo loads models from `data/sample/models` inside the bundle; no external model hosting is needed.
- If you want the Space to auto-sync from this GitHub repo instead of manual pushes, use the "Link GitHub repository" option in the Space settings.
