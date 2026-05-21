# mistral2api

Convert [chat.mistral.ai](https://chat.mistral.ai/) web chat into OpenAI-compatible API.

---

> ## ⚠️ DISCLAIMER / 免责声明
>
> ### 🌍 Foreign Website Notice / 外国网站提示
> **This project interfaces with a foreign website (`chat.mistral.ai`). Users must ensure their network can access this site. The author is not responsible for any network access issues.**
>
> **本项目对接的是外国网站 (`chat.mistral.ai`)，请确保您的网络可以访问该网站。作者不对任何网络访问问题负责。**
>
> ### ⚖️ Legal Disclaimer / 法律免责
> **This project is for educational and research purposes only. Commercial or illegal use is strictly prohibited.**
>
> - This project reverse-engineers the internal web API of chat.mistral.ai and is **NOT an official Mistral AI API**.
> - Using this project may violate [Mistral AI Terms of Service](https://mistral.ai/legal/terms/). Users assume all risks.
> - This project does not store, collect, or share any user data.
> - The author is not responsible for any consequences including account bans, data loss, or legal issues.
> - This project is provided "as is" without any warranty. Use at your own risk.
>
> **本项目仅供学习和研究使用，严禁用于任何商业或非法目的。**
>
> - 本项目通过逆向工程 chat.mistral.ai 网页 API 实现，并非 Mistral AI 官方 API。
> - 使用本项目可能违反 Mistral AI 服务条款，使用者需自行承担所有风险。
> - 本项目不存储、收集或共享任何用户数据。
> - 作者不对因使用本项目导致的任何后果负责，包括但不限于账号封禁、数据丢失、法律纠纷等。
> - 本项目按"原样"提供，不做任何保证，使用者需自行承担风险。

---

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Streaming (SSE) and non-streaming support
- `/v1/models` model list endpoint
- **Guest mode**: No login required, uses chat.mistral.ai anonymous chat
- **Cookie mode**: Login-based session for enhanced access
- Simple Web management UI (MiMo2API-style)
- Multi-model support
- cURL/Cookie import for session management

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start server

```bash
python main.py
```

Default: `http://localhost:8000`

### 3. Configuration (optional)

Create `config.json`:

```json
{
  "api_keys": ["sk-mistral2api"],
  "host": "0.0.0.0",
  "port": 8000,
  "default_model": "mistral-large-latest",
  "request_timeout": 120
}
```

Or use environment variables:

```bash
set PORT=8000
set API_KEYS=sk-mistral2api
set DEFAULT_MODEL=mistral-large-latest
```

## API Usage

### List Models

```bash
curl http://localhost:8000/v1/models -H "Authorization: Bearer sk-mistral2api"
```

### Chat Completion (Non-streaming)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-mistral2api" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"mistral-large-latest\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

### Streaming Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-mistral2api" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"mistral-large-latest\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a story\"}],\"stream\":true}"
```

### Supported Models

- `mistral-large-latest`
- `mistral-small-latest`
- `mistral-medium-latest`
- `mistral-8b-latest`
- `pixtral-large-latest`
- `codestral-latest`

## Project Structure

```
mistral2api/
├── main.py                # Entry point
├── requirements.txt       # Dependencies
├── config.json            # Configuration (optional)
├── cookies.json           # Cookie storage (auto-generated)
├── README.md
└── app/
    ├── routes.py          # OpenAI-compatible + admin endpoints
    ├── mistral_client.py  # Mistral Chat API client (guest + cookie mode)
    ├── models.py          # Pydantic models (OpenAI format)
    ├── config.py          # Configuration + cookie manager
    └── web/
        └── index.html     # Web management UI (MiMo2API-style)
```

## How It Works

1. Client sends OpenAI-format request to `/v1/chat/completions`
2. mistral2api converts messages to Mistral Chat format
3. **Guest mode**: Automatically visits chat.mistral.ai to establish session → calls API
4. **Cookie mode**: Uses manually imported cookies (from cURL or browser DevTools)
5. Creates chat via `chat.mistral.ai/api/trpc/message.newChat`
6. Streams response from `chat.mistral.ai/api/chat` (custom SSE format `0:"text"`)
7. Converts tokens back to OpenAI SSE format

## Cookie Mode Setup

If guest mode fails (e.g. network restrictions or Mistral requires login):

1. Open `https://chat.mistral.ai/chat` in your browser
2. Open DevTools → Network tab → send a message
3. Copy the `/api/chat` request as cURL
4. Paste cURL into the WebUI's "Cookie管理" tab → click "解析" → "保存"

Or manually input cookies:
- `ory_session_*`: Ory Kratos session cookie
- `csrftoken`: CSRF protection token

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [MiMo2API](https://github.com/Fly143/MiMo2API) - Architecture reference
- [ChatALL](https://github.com/ai-shifu/ChatALL) - Mistral API reverse engineering reference
- [Mistral AI](https://mistral.ai) - The amazing AI models
