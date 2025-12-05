# server/tools/mobile_calendar_auth.py
"""
Google Calendar authentication designed for mobile app backends.
Provides OAuth flow suitable for mobile apps calling backend APIs.
"""
import os
import pickle
import json
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/calendar']

class MobileCalendarAuth:
    def __init__(self):
        self.credentials_file = "server/client_secret_181769508623-jgasa2nmi4hlckgmiq9mv8i4l0ijo6f9.apps.googleusercontent.com.json"
        self.token_file = "token.pickle"
        self.creds = None
        
    def get_authorization_url(self):
        """Get authorization URL for mobile app to redirect user to"""
        try:
            flow = Flow.from_client_secrets_file(
                self.credentials_file, 
                scopes=SCOPES,
                redirect_uri="https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net/oauth/callback"
            )
            
            authorization_url, state = flow.authorization_url(
                access_type='offline',  # Enables refresh tokens
                include_granted_scopes='true'
            )
            
            # Store only the essential data we need to recreate the flow
            # Read the client secrets file directly to save it
            with open(self.credentials_file, 'r') as f:
                client_secrets = json.load(f)
            
            flow_data = {
                'client_secrets': client_secrets,
                'scopes': SCOPES,
                'redirect_uri': "https://virtual-assistent-cudwb7h9e6avdkfu.eastus-01.azurewebsites.net/oauth/callback",
                'state': state
            }
            
            with open('oauth_state.json', 'w') as f:
                json.dump(flow_data, f)
            
            return {
                'authorization_url': authorization_url,
                'state': state
            }
            
        except Exception as e:
            print(f"Error creating authorization URL: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def handle_oauth_callback(self, authorization_code, state):
        """Handle OAuth callback from mobile app"""
        try:
            # Load stored flow data
            with open('oauth_state.json', 'r') as f:
                flow_data = json.load(f)
            
            stored_state = flow_data['state']
            
            # Verify state for security
            if state != stored_state:
                raise ValueError("State mismatch - potential security issue")
            
            # Create new flow with the stored client secrets
            flow = Flow.from_client_config(
                flow_data['client_secrets'],
                scopes=flow_data['scopes'],
                redirect_uri=flow_data['redirect_uri']
            )
            
            # Exchange authorization code for tokens
            flow.fetch_token(code=authorization_code)
            self.creds = flow.credentials
            
            # Save credentials
            with open(self.token_file, 'wb') as token:
                pickle.dump(self.creds, token)
            
            # Clean up state file
            if os.path.exists('oauth_state.json'):
                os.remove('oauth_state.json')
            
            return True
            
        except Exception as e:
            print(f"Error handling OAuth callback: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_service_for_backend(self):
        """Get Google Calendar service for backend operations"""
        # Check if we have saved credentials
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                self.creds = pickle.load(token)
        
        # Refresh if needed
        if self.creds and self.creds.expired and self.creds.refresh_token:
            try:
                self.creds.refresh(Request())
                # Save refreshed credentials
                with open(self.token_file, 'wb') as token:
                    pickle.dump(self.creds, token)
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                return None
        
        if self.creds and self.creds.valid:
            return build('calendar', 'v3', credentials=self.creds)
        
        return None

# Global instance for mobile backend
_mobile_auth = MobileCalendarAuth()

def get_mobile_auth_url():
    """Get authorization URL for mobile app"""
    return _mobile_auth.get_authorization_url()

def handle_mobile_auth_callback(code, state):
    """Handle OAuth callback from mobile app"""
    return _mobile_auth.handle_oauth_callback(code, state)

def get_calendar_service_mobile():
    """Get Google Calendar service for mobile backend"""
    return _mobile_auth.get_service_for_backend()

def is_mobile_authenticated():
    """Check if mobile backend has valid credentials"""
    if not os.path.exists(_mobile_auth.token_file):
        print("📅 No token file found")
        return False
    
    try:
        # Try to get the service to verify credentials are valid
        service = _mobile_auth.get_service_for_backend()
        if service is None:
            print("📅 Failed to get calendar service - credentials may be expired")
            return False
        
        # Try a simple API call to verify the service works
        service.calendarList().list(maxResults=1).execute()
        print("📅 Authentication verified successfully")
        return True
    except Exception as e:
        print(f"📅 Authentication check failed: {e}")
        # Remove invalid token file
        if os.path.exists(_mobile_auth.token_file):
            os.remove(_mobile_auth.token_file)
            print("📅 Removed invalid token file")
        return False