# Setup

Requires Python 3.11+ (developed on 3.13).

```bash
cd ~/f1-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

FastF1 downloads race data on first use and caches it in `./cache` (ignored by git).
Raw data goes to `data/raw/` (also ignored); generated files go to `outputs/`.
