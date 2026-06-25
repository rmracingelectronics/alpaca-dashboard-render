web: gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
worker: python trading_worker.py
