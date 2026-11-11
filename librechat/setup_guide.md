# LibreChat Integration Guide for SpecRoute Hybrid Speculative Decoding

This guide explains how to connect your **LibreChat** interface to the local **SpecRoute** proxy engine.

---

## 1. Architecture Flow

```
[ User in Browser ] 
        │
        ▼ (Port 3080)
[ LibreChat UI ]
        │
        ▼ (OpenAI SSE Stream /v1/chat/completions)
[ Local SpecRoute Proxy (Port 8000) ]
        │
        ├── [ Local Draft Model (0.5B / 1.5B) ] ── (Fast lookahead guessing)
        │
        ▼ (WebSocket / HTTP Verify)
[ Remote Colab / Cloud GPU Target Model (7B / 8B) ] ── (Single-pass parallel verification)
```

---

## 2. Quick Setup

### Option A: If running LibreChat with Docker

1. Copy `librechat/librechat.yaml` to your LibreChat repository folder as `librechat.yaml`.
2. Copy `librechat/docker-compose.override.yml` to your LibreChat folder as `docker-compose.override.yml`.
3. In `librechat.yaml`, ensure the `baseURL` is set to:
   ```yaml
   baseURL: "http://host.docker.internal:8000/v1"
   ```
4. Start or restart LibreChat:
   ```bash
   docker compose up -d
   ```

### Option B: If running LibreChat natively (Node.js)

1. Copy `librechat/librechat.yaml` to your LibreChat root directory.
2. In `librechat.yaml`, select or use:
   ```yaml
   baseURL: "http://127.0.0.1:8000/v1"
   ```
3. Start LibreChat:
   ```bash
   npm run backend
   ```

---

## 3. Starting the System

### Step 1: Start the Remote Target Server
- If using Google Colab or Kaggle: Open `notebooks/colab_kaggle_target_server.ipynb`, select T4/L4 GPU runtime, run all cells, and copy the public ngrok URL.
- If testing locally: Run:
  ```bash
  python run_server.py --simulate --port 8001
  ```

### Step 2: Configure Remote URL
Edit `config/config.yaml`:
```yaml
remote_server:
  base_url: "https://your-ngrok-subdomain.ngrok-free.app"  # Or http://127.0.0.1:8001
```

### Step 3: Start the Local SpecRoute Proxy
Run:
```bash
python run_proxy.py --port 8000
```

### Step 4: Open LibreChat
1. Navigate to `http://localhost:3080` in your browser.
2. In the model selector dropdown at the top, select **SpecRoute Hybrid (Local 1.5B + Cloud 7B)**.
3. Start chatting! You will observe:
   - Ultra-fast token streaming.
   - Smooth typewriter typing without jitter.
   - Live speculative acceptance rate logs in your terminal console:
     ```text
     [Speculation] Step 1: 4/4 tokens accepted (100.0%) | Corrective: ' how' | Cumulative: 4/4 (100.0%) | Latency: 28.4ms
     [Speculation] Step 2: 3/4 tokens accepted (75.0%)  | Corrective: ' can' | Cumulative: 7/8 (87.5%)  | Latency: 25.1ms
     ```

