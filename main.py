import os
import uvicorn
from server.mcp_server import app

if __name__ == "__main__":
    # Get port from environment variable (Azure sets PORT) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # For Azure deployment, bind to 0.0.0.0 and use environment port
    uvicorn.run(
        "server.mcp_server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )