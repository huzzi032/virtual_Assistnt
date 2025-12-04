# Voice Assistant Backend

Production-ready backend for voice assistant with Azure Speech-to-Text, OpenAI integration, Google Calendar, and Zoom integration.

## Features
- Speech-to-text transcription using GPT-4o
- LLM-powered conversation analysis
- Todo management and task extraction
- Google Calendar integration
- Zoom meeting integration with webhooks
- Real-time audio processing

## Deployment to Azure

### Prerequisites
- Azure account
- GitHub account
- Azure CLI installed (optional)

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Backend ready for deployment"
   git push origin main
   ```

2. **Azure Portal Setup**
   - Go to Azure Portal (https://portal.azure.com)
   - Create a new "Web App"
   - Choose Python 3.11 as runtime
   - Connect to your GitHub repository
   - Select this repository

3. **Configure Environment Variables in Azure**
   Go to Configuration → Application Settings and add:
   ```
   OPENAI_API_KEY=your_openai_key
   OPENAI_BASE_URL=your_openai_base_url
   AZURE_SPEECH_KEY=your_azure_speech_key
   AZURE_SPEECH_REGION=your_azure_region
   ZOOM_CLIENT_ID=your_zoom_client_id
   ZOOM_CLIENT_SECRET=your_zoom_client_secret
   ZOOM_WEBHOOK_SECRET=your_zoom_webhook_secret
   ```

4. **Set Startup Command**
   In Azure Portal → Configuration → General Settings → Startup Command:
   ```
   python main.py
   ```

5. **Deploy**
   - Azure will automatically deploy from GitHub
   - Monitor deployment in Deployment Center

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with your credentials

3. Run server:
   ```bash
   python main.py
   ```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/process-audio` - Process audio file
- `GET /api/auth/google/url` - Get Google OAuth URL
- `GET /api/auth/status` - Check authentication status
- `GET /api/zoom/auth/url` - Get Zoom OAuth URL
- `POST /api/zoom/meetings/detect` - Detect Zoom meetings
- `POST /webhooks/zoom` - Zoom webhook handler

## Support
For issues or questions, check the server logs in Azure Portal.
