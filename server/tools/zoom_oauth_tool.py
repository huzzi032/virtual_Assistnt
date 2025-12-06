# server/tools/zoom_oauth_tool.py
"""
Zoom OAuth Integration Tool
Handles Zoom OAuth flow for meeting access and real-time transcription
"""

import os
import json
import aiohttp
import asyncio
from urllib.parse import urlencode
from dotenv import load_dotenv
import sqlite3
from datetime import datetime, timedelta
import secrets
import hashlib
import base64

load_dotenv()

class ZoomOAuthManager:
    def __init__(self):
        """Initialize Zoom OAuth configuration"""
        
        # Zoom OAuth credentials from environment
        self.client_id = os.getenv('ZOOM_CLIENT_ID')
        self.client_secret = os.getenv('ZOOM_CLIENT_SECRET')
        # Use Zoom Connector OAuth flow
        self.connector_auth_url = 'https://integrations.zoom.us/connectors/oauth/KSrs7u0yQXihzRv_qi0ACg/bef_authorization'
        self.redirect_uri = 'https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net/api/zoom/auth/callback'
        
        # Zoom API endpoints
        self.auth_base_url = "https://zoom.us/oauth"
        self.api_base_url = "https://api.zoom.us/v2"
        
        # OAuth scopes for meeting access and real-time features
        self.scopes = [
            'meeting:read',           # Read meeting info
            'meeting:write',          # Create/modify meetings
            'webinar:read',           # Read webinar info  
            'recording:read',         # Access recordings
            'meeting:master',         # Meeting management
            'user:read'               # Basic user info
        ]
        
        # Database connection for storing tokens
        self.db_path = "database.db"
        self._init_database()
        
        print(f"🔗 Zoom OAuth initialized")
        print(f"   Client ID: {self.client_id[:10]}..." if self.client_id else "   ❌ Missing ZOOM_CLIENT_ID")
        print(f"   Redirect URI: {self.redirect_uri}")
        print(f"   Scopes: {', '.join(self.scopes)}")

    def _init_database(self):
        """Initialize database table for storing Zoom tokens"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Create zoom_tokens table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zoom_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT DEFAULT 'default_user',
                    access_token TEXT NOT NULL,
                    refresh_token TEXT,
                    token_type TEXT DEFAULT 'Bearer',
                    expires_at DATETIME NOT NULL,
                    scope TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create zoom_meetings table for tracking meetings
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zoom_meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id TEXT UNIQUE NOT NULL,
                    meeting_uuid TEXT,
                    topic TEXT,
                    join_url TEXT,
                    start_url TEXT,
                    start_time DATETIME,
                    duration INTEGER,
                    status TEXT DEFAULT 'scheduled',
                    transcription_enabled BOOLEAN DEFAULT 0,
                    real_time_processing BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ Zoom database tables initialized")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")

    def get_authorization_url(self, state: str | None = None) -> dict:
        """Generate Zoom Connector OAuth authorization URL"""
        
        if not self.client_id:
            return {
                "success": False,
                "error": "ZOOM_CLIENT_ID not configured in environment variables"
            }
        
        # Generate state for security if not provided
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Build standard OAuth parameters for the callback URL
        oauth_params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes),
            'state': state
        }
        
        # Create the standard OAuth URL that will be used as callback
        oauth_callback_url = f"{self.auth_base_url}/authorize?" + urlencode(oauth_params)
        
        # Build the Zoom Connector authorization URL with the callback
        connector_params = {
            'call_back_url': oauth_callback_url
        }
        
        # Final Zoom Connector authorization URL
        authorization_url = f"{self.connector_auth_url}?" + urlencode(connector_params)
        
        print(f"🔗 Generated Zoom Connector auth URL with state: {state[:10]}...")
        
        return {
            "success": True,
            "authorization_url": authorization_url,
            "state": state,
            "redirect_uri": self.redirect_uri
        }

    async def exchange_code_for_tokens(self, code: str, state: str | None = None) -> dict:
        """Exchange authorization code for access tokens"""
        
        if not self.client_id or not self.client_secret:
            return {
                "success": False,
                "error": "Zoom OAuth credentials not configured"
            }
        
        # Prepare token exchange request for Zoom connector
        token_url = "https://integrations.zoom.us/connectors/oauth/7Ua1f1h_SiGDCn1gYfJqAQ/token"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri
        }
        
        try:
            print(f"🔄 Exchanging code for Zoom tokens...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, headers=headers, json=data) as response:
                    
                    if response.status == 200:
                        tokens = await response.json()
                        print(f"✅ Zoom tokens received successfully")
                        
                        # Store tokens in database
                        await self._store_tokens(tokens)
                        
                        return {
                            "success": True,
                            "access_token": tokens.get('access_token'),
                            "token_type": tokens.get('token_type', 'Bearer'),
                            "expires_in": tokens.get('expires_in'),
                            "scope": tokens.get('scope')
                        }
                    else:
                        error_text = await response.text()
                        print(f"❌ Token exchange failed: {response.status} - {error_text}")
                        
                        return {
                            "success": False,
                            "error": f"Token exchange failed: {error_text}"
                        }
                        
        except Exception as e:
            print(f"❌ Token exchange error: {e}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}"
            }

    async def _store_tokens(self, tokens: dict):
        """Store access tokens in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Calculate expiration time
            expires_in = tokens.get('expires_in', 3600)  # Default 1 hour
            expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            # Delete existing tokens for this user
            cur.execute("DELETE FROM zoom_tokens WHERE user_id = ?", ('default_user',))
            
            # Insert new tokens
            cur.execute("""
                INSERT INTO zoom_tokens 
                (user_id, access_token, refresh_token, token_type, expires_at, scope)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                'default_user',
                tokens.get('access_token'),
                tokens.get('refresh_token'),
                tokens.get('token_type', 'Bearer'),
                expires_at,
                tokens.get('scope')
            ))
            
            conn.commit()
            conn.close()
            
            print(f"💾 Zoom tokens stored, expires at: {expires_at}")
            
        except Exception as e:
            print(f"❌ Error storing tokens: {e}")

    def is_authenticated(self) -> dict:
        """Check if user is authenticated with Zoom"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT access_token, expires_at, scope 
                FROM zoom_tokens 
                WHERE user_id = ? AND expires_at > datetime('now')
                ORDER BY created_at DESC LIMIT 1
            """, ('default_user',))
            
            result = cur.fetchone()
            conn.close()
            
            if result:
                access_token, expires_at, scope = result
                return {
                    "authenticated": True,
                    "expires_at": expires_at,
                    "scope": scope,
                    "message": "Ready for Zoom meeting integration"
                }
            else:
                return {
                    "authenticated": False,
                    "message": "Zoom authentication required"
                }
                
        except Exception as e:
            print(f"❌ Auth check error: {e}")
            return {
                "authenticated": False,
                "message": f"Authentication check failed: {str(e)}"
            }

    async def get_valid_access_token(self) -> str:
        """Get valid access token, refreshing if necessary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT access_token, refresh_token, expires_at 
                FROM zoom_tokens 
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, ('default_user',))
            
            result = cur.fetchone()
            conn.close()
            
            if not result:
                raise Exception("No Zoom tokens found - authentication required")
            
            access_token, refresh_token, expires_at_str = result
            
            # Check if token is still valid (with 5 minute buffer)
            expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
            buffer_time = datetime.now() + timedelta(minutes=5)
            
            if expires_at > buffer_time:
                return access_token
            
            # Token expired, try to refresh
            if refresh_token:
                print("🔄 Refreshing expired Zoom access token...")
                new_tokens = await self._refresh_access_token(refresh_token)
                
                if new_tokens.get('success'):
                    return new_tokens.get('access_token', '')
                else:
                    raise Exception("Token refresh failed - re-authentication required")
            else:
                raise Exception("No refresh token available - re-authentication required")
                
        except Exception as e:
            print(f"❌ Access token error: {e}")
            raise e

    async def _refresh_access_token(self, refresh_token: str) -> dict:
        """Refresh access token using refresh token"""
        
        token_url = f"{self.auth_base_url}/token"
        
        # Basic auth header
        credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        
        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, headers=headers, data=data) as response:
                    
                    if response.status == 200:
                        tokens = await response.json()
                        await self._store_tokens(tokens)
                        
                        return {
                            "success": True,
                            "access_token": tokens.get('access_token')
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Refresh failed: {error_text}"
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"Refresh error: {str(e)}"
            }

# Global instance
zoom_oauth_manager = ZoomOAuthManager()

# Convenience functions for external use
def get_zoom_auth_url(state: str | None = None) -> dict:
    """Get Zoom OAuth authorization URL"""
    return zoom_oauth_manager.get_authorization_url(state)

async def handle_zoom_oauth_callback(code: str, state: str | None = None) -> dict:
    """Handle Zoom OAuth callback"""
    return await zoom_oauth_manager.exchange_code_for_tokens(code, state)

def is_zoom_authenticated() -> dict:
    """Check Zoom authentication status"""
    return zoom_oauth_manager.is_authenticated()

async def get_zoom_access_token() -> str:
    """Get valid Zoom access token"""
    return await zoom_oauth_manager.get_valid_access_token()

if __name__ == "__main__":
    # Test OAuth flow
    manager = ZoomOAuthManager()
    
    # Generate auth URL
    auth_data = manager.get_authorization_url()
    print(f"Auth URL: {auth_data}")
    
    # Check authentication status
    status = manager.is_authenticated()
    print(f"Auth Status: {status}")