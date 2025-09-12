# FastAPI LocalGPT2 Server - Railway Deployment

This is a FastAPI server that provides AI-powered service provider search using OpenAI's GPT-4o model.

## Quick Railway Deployment

### 1. Push to GitHub
```bash
git add .
git commit -m "Add Railway deployment config"
git push origin main
```

### 2. Deploy to Railway
1. Go to [Railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository and the `localgpt2` folder
5. Railway will automatically detect the Python project

### 3. Set Environment Variables
In Railway dashboard, go to your project → Variables tab and add:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 4. Deploy!
Railway will automatically build and deploy your app. You'll get a public URL like:
`https://yourapp-production.up.railway.app`

## API Endpoints

- `GET /` - Health check
- `GET /health` - Health status
- `POST /api/query` - Main query endpoint

### Example Usage
```bash
curl -X POST "https://yourapp-production.up.railway.app/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "plumber near Lahore"}'
```

## Features
- GPT-4o powered service provider generation
- Pakistani business data with realistic addresses and phone numbers
- Usage analytics with token costs
- CORS enabled for web frontend integration
- Automatic scaling on Railway

## Environment Variables
- `OPENAI_API_KEY` (required) - Your OpenAI API key
- `PORT` (auto-set by Railway) - Server port

## Cost Monitoring
The server includes detailed usage analytics with current OpenAI pricing:
- Input tokens: $5.00 per 1M tokens
- Output tokens: $15.00 per 1M tokens

All costs are calculated and logged for monitoring.
