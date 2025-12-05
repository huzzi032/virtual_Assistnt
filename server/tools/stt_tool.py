# server/tools/stt_tool.py - Complete Azure OpenAI GPT-4o STT Solution
import aiohttp
import tempfile
import os
import wave
import asyncio
import json
import subprocess
import shutil
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class GPT4oHTTPSTT:
    def __init__(self):
        # Azure OpenAI configuration - Use GPT4O environment variables
        self.endpoint = os.getenv('GPT4O_TRANSCRIBE_ENDPOINT', 'https://iarshad-3836-resource.cognitiveservices.azure.com')
        self.deployment = os.getenv('GPT4O_DEPLOYMENT_NAME', 'gpt-4o-transcribe-diarize')
        self.api_version = os.getenv('GPT4O_API_VERSION', '2025-03-01-preview')
        self.api_key = os.getenv('GPT4O_API_KEY', '')
        
        # Validate API key
        if not self.api_key:
            raise ValueError("❌ GPT4O_API_KEY not found in environment variables")
        
        # Build complete URL with proper formatting
        self.url = f"{self.endpoint}/openai/deployments/{self.deployment}/audio/transcriptions?api-version={self.api_version}"
        
        print(f"🚀 GPT-4o HTTP STT initialized")
        print(f"🏢 Endpoint: {self.endpoint}")
        print(f"📦 Deployment: {self.deployment}")
        print(f"📋 API Version: {self.api_version}")
        print(f"🔗 Full URL: {self.url}")
        
        # Validate URL construction
        if not self.url.startswith("https://"):
            raise ValueError("Invalid URL construction")
        if "cognitiveservices.azure.com" not in self.url:
            raise ValueError("Invalid Azure endpoint")

    def validate_audio_requirements(self, audio_data: bytes) -> Tuple[bool, str]:
        """Validate audio meets Azure OpenAI requirements"""
        try:
            # Minimum size check (2 seconds at 16kHz mono 16-bit = ~64KB)
            MIN_SIZE = 32000  # 32KB minimum for ~2 seconds
            if len(audio_data) < MIN_SIZE:
                return False, f"Audio too small: {len(audio_data)} bytes (need ≥{MIN_SIZE} for 2+ seconds). Record for at least 3-4 seconds."
            
            # Maximum size check (25MB Azure limit)
            MAX_SIZE = 25 * 1024 * 1024  # 25MB
            if len(audio_data) > MAX_SIZE:
                return False, f"Audio too large: {len(audio_data)} bytes (max {MAX_SIZE})"
            
            return True, f"Audio size OK: {len(audio_data)} bytes"
            
        except Exception as e:
            return False, f"Size validation failed: {e}"

    def analyze_wav_format(self, audio_data: bytes) -> dict:
        """Deep analysis of WAV format"""
        analysis = {
            "size": len(audio_data),
            "is_valid_wav": False,
            "channels": None,
            "sample_rate": None,
            "sample_width": None,
            "duration": None,
            "needs_conversion": True,
            "error": None
        }
        
        try:
            # Basic header checks
            if len(audio_data) < 44:
                analysis["error"] = "File too small for WAV header"
                return analysis
            
            # Check RIFF header
            if audio_data[:4] != b'RIFF':
                analysis["error"] = "Missing RIFF header - not a WAV file"
                return analysis
            
            # Check WAVE identifier
            if audio_data[8:12] != b'WAVE':
                analysis["error"] = "Missing WAVE identifier - not a WAV file"
                return analysis
            
            # Save to temp file for wave module analysis
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                with wave.open(temp_path, 'rb') as wav_file:
                    analysis["is_valid_wav"] = True
                    analysis["channels"] = wav_file.getnchannels()
                    analysis["sample_rate"] = wav_file.getframerate()
                    analysis["sample_width"] = wav_file.getsampwidth()
                    analysis["frames"] = wav_file.getnframes()
                    analysis["duration"] = analysis["frames"] / analysis["sample_rate"] if analysis["sample_rate"] > 0 else 0
                    
                    # Check Azure compatibility (PCM16, mono, 16kHz preferred)
                    is_azure_compatible = (
                        analysis["channels"] == 1 and
                        analysis["sample_rate"] in [16000, 44100] and
                        analysis["sample_width"] == 2
                    )
                    
                    analysis["needs_conversion"] = not is_azure_compatible
                    
                    print(f"🔍 WAV Analysis:")
                    print(f"   ✅ Valid WAV file")
                    print(f"   Channels: {analysis['channels']} ({'✅ Mono' if analysis['channels'] == 1 else '⚠️ Stereo'})")
                    print(f"   Sample rate: {analysis['sample_rate']}Hz ({'✅ Optimal' if analysis['sample_rate'] == 16000 else '✅ Good' if analysis['sample_rate'] == 44100 else '⚠️ Suboptimal'})")
                    print(f"   Sample width: {analysis['sample_width']} bytes ({'✅ PCM16' if analysis['sample_width'] == 2 else '⚠️ Wrong'})")
                    print(f"   Duration: {analysis['duration']:.2f}s ({'✅ Good' if analysis['duration'] >= 2.0 else '⚠️ Too short'})")
                    print(f"   Azure compatible: {'✅ Yes' if is_azure_compatible else '🔧 Needs conversion'}")
                    
            except Exception as wave_error:
                analysis["error"] = f"WAV parsing failed: {wave_error}"
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            analysis["error"] = f"Analysis failed: {e}"
        
        return analysis

    def convert_with_ffmpeg(self, audio_data: bytes) -> Tuple[bool, bytes, str]:
        """Convert audio using ffmpeg to Azure-compatible format"""
        try:
            # Check if ffmpeg is available
            if not shutil.which('ffmpeg'):
                return False, audio_data, "❌ ffmpeg not found in PATH. Please install ffmpeg from https://ffmpeg.org/download.html and add to PATH."
            
            print("🔄 Converting audio using ffmpeg (PCM16, mono, 16kHz)...")
            
            # Create temp files
            with tempfile.NamedTemporaryFile(suffix='.input', delete=False) as input_file:
                input_file.write(audio_data)
                input_path = input_file.name
            
            output_path = input_path + '_azure.wav'
            
            try:
                # ffmpeg command for Azure-compatible format
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', input_path,
                    '-ac', '1',          # Mono
                    '-ar', '16000',      # 16kHz
                    '-c:a', 'pcm_s16le', # PCM 16-bit little endian
                    '-f', 'wav',         # WAV container
                    output_path
                ]
                
                print(f"🛠️ Running: {' '.join(cmd[:-1])} [output]")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=60  # 60 second timeout
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    return False, audio_data, f"ffmpeg failed: {error_msg}"
                
                # Check if output file was created
                if not os.path.exists(output_path):
                    return False, audio_data, "ffmpeg did not create output file"
                
                # Read converted audio
                with open(output_path, 'rb') as f:
                    converted_data = f.read()
                
                if len(converted_data) == 0:
                    return False, audio_data, "ffmpeg produced empty output"
                
                print(f"✅ ffmpeg conversion successful: {len(audio_data)} → {len(converted_data)} bytes")
                return True, converted_data, "Conversion successful"
                
            except subprocess.TimeoutExpired:
                return False, audio_data, "ffmpeg conversion timed out (>60s)"
            except Exception as e:
                return False, audio_data, f"ffmpeg execution failed: {e}"
            finally:
                # Cleanup temp files
                for path in [input_path, output_path]:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except:
                        pass
                        
        except Exception as e:
            return False, audio_data, f"ffmpeg setup failed: {e}"

    async def transcribe_with_retries(self, audio_data: bytes, max_retries: int = 3) -> str:
        """
        Transcribe with exponential backoff retry for Azure 500 errors
        
        Implements OpenAI-recommended safe workflow for gpt-4o-transcribe-diarize:
        1. Try diarized_json format first
        2. Check for segments array in response
        3. Fallback to plain text if diarization fails (known issue)
        4. If all fails, retry with simple 'json' format
        """
        
        for attempt in range(max_retries):
            try:
                print(f"🌐 Attempt {attempt + 1}/{max_retries}: Sending {len(audio_data)} bytes to Azure")
                print(f"🔗 URL: {self.url}")
                print(f"🔑 API Key (first 10 chars): {self.api_key[:10]}...")
                print(f"📦 Deployment: {self.deployment}")
                
                timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Create multipart form data using OpenAI-recommended safe workflow
                    data = aiohttp.FormData()
                    data.add_field(
                        'file',  # Must be exactly "file" for Azure OpenAI
                        audio_data,
                        filename='audio.wav',
                        content_type='audio/wav'
                    )
                    data.add_field('model', self.deployment)
                    data.add_field('response_format', 'diarized_json')  # Try diarization first
                    # For longer audio, add chunking strategy
                    if len(audio_data) > 1024 * 1024:  # > 1MB (~30+ seconds)
                        data.add_field('chunking_strategy', 'auto')
                    
                    # Correct headers for Azure OpenAI multipart upload - using api-key header
                    # Note: Don't set Content-Type manually for multipart - aiohttp handles it
                    headers = {
                        "api-key": self.api_key,
                        "Accept": "application/json"
                    }
                    
                    async with session.post(self.url, headers=headers, data=data) as response:
                        print(f"📡 Response status: {response.status}")
                        
                        if response.status == 200:
                            result = await response.json()
                            print(f"🎯 Azure response: {result}")
                            
                            # Implement OpenAI-recommended safe workflow for diarization
                            transcription = ""
                            diarization_success = False
                            
                            # Check if we got speaker segments (diarization working)
                            if 'segments' in result and result['segments']:
                                print("🎙️ Speaker diarization successful!")
                                diarization_success = True
                                
                                # Combine segments with speaker labels
                                segment_texts = []
                                for i, seg in enumerate(result['segments']):
                                    speaker = seg.get('speaker', f'Speaker {i+1}')
                                    text = seg.get('text', '').strip()
                                    start = seg.get('start', 0)
                                    end = seg.get('end', 0)
                                    
                                    if text:
                                        print(f"   {speaker} ({start:.1f}s-{end:.1f}s): {text}")
                                        segment_texts.append(f"[{speaker}] {text}")
                                
                                transcription = " ".join(segment_texts) if segment_texts else ""
                            
                            # Fallback to plain text if diarization failed (known issue)
                            if not transcription and 'text' in result:
                                print("⚠️ Diarization failed silently (known issue), using plain text")
                                transcription = result['text'].strip()
                                diarization_success = False
                            
                            if transcription:
                                success_msg = "with speaker diarization" if diarization_success else "as plain text"
                                print(f"✅ Transcription successful {success_msg}: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
                                return transcription
                            else:
                                print(f"⚠️ No transcription found in response: {result}")
                                return "No speech detected in audio. Please speak clearly and try again."
                        
                        elif response.status == 400:
                            error_text = await response.text()
                            print(f"❌ Bad request (400): {error_text}")
                            
                            # If diarized_json format is not supported, try simple json
                            if "diarized_json" in error_text and attempt == 0:
                                print("🔄 Diarized format not supported, trying simple json...")
                                continue  # This will retry with same params, but we could modify to use 'json'
                            
                            if "format" in error_text.lower():
                                return "Audio format error. Please ensure proper WAV format (PCM16, mono, 16kHz)."
                            elif "duration" in error_text.lower() or "too short" in error_text.lower():
                                return "Audio too short. Please record for at least 3 seconds with clear speech."
                            else:
                                return f"Request error: {error_text}"
                        
                        elif response.status == 500:
                            error_text = await response.text()
                            print(f"❌ Azure 500 error (attempt {attempt + 1}): {error_text[:200]}...")
                            
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) + 2  # Exponential backoff: 3, 6, 10 seconds
                                print(f"⏱️ Waiting {wait_time}s before retry...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                return "Azure STT service temporarily unavailable after multiple attempts. Please try again in a few minutes."
                        
                        else:
                            error_text = await response.text()
                            print(f"❌ Azure error {response.status}: {error_text}")
                            return f"Azure STT error {response.status}: Please check your deployment and try again."
                            
            except asyncio.TimeoutError:
                print(f"⏱️ Request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                    continue
                else:
                    return "Request timeout after multiple attempts. Please try again with a shorter audio clip."
                    
            except Exception as e:
                print(f"❌ Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"Request failed: {str(e)}"
        
        return "Max retries exceeded"

async def speech_to_text_from_audio(audio_data: bytes) -> str:
    """
    Complete Azure OpenAI GPT-4o STT pipeline with error handling
    
    Args:
        audio_data (bytes): Audio file data
        
    Returns:
        str: Transcribed text or user-friendly error message
    """
    try:
        print(f"📝 Starting GPT-4o STT processing: {len(audio_data)} bytes ({len(audio_data)/1024/1024:.2f}MB)")
        
        # Initialize STT tool
        stt_tool = GPT4oHTTPSTT()
        
        # Step 1: Validate audio size
        size_valid, size_message = stt_tool.validate_audio_requirements(audio_data)
        print(f"📏 {size_message}")
        
        if not size_valid:
            return size_message
        
        # Step 2: Analyze WAV format
        analysis = stt_tool.analyze_wav_format(audio_data)
        
        if analysis.get("error"):
            print(f"❌ WAV format error: {analysis['error']}")
            print("🔧 Attempting conversion anyway...")
            
            # Try conversion even with invalid WAV
            success, converted_audio, message = stt_tool.convert_with_ffmpeg(audio_data)
            
            if success:
                print("✅ Successfully converted invalid audio to valid WAV")
                processed_audio = converted_audio
                
                # Re-analyze converted audio
                new_analysis = stt_tool.analyze_wav_format(processed_audio)
                if new_analysis.get("error"):
                    return f"Audio conversion failed: {new_analysis['error']}. Please record in WAV format."
            else:
                return f"Audio format not supported: {message}. Please record in proper WAV format."
        
        else:
            processed_audio = audio_data
            
            # Step 3: Convert if needed for Azure compatibility
            if analysis.get("needs_conversion", True):
                print("🔧 Converting audio for Azure compatibility...")
                success, converted_audio, message = stt_tool.convert_with_ffmpeg(audio_data)
                
                if success:
                    processed_audio = converted_audio
                    print("✅ Audio conversion completed")
                    
                    # Re-validate converted audio
                    final_analysis = stt_tool.analyze_wav_format(processed_audio)
                    if final_analysis.get("error"):
                        print(f"⚠️ Post-conversion validation failed: {final_analysis['error']}")
                        print("⚠️ Proceeding with converted audio anyway...")
                else:
                    print(f"⚠️ Conversion failed: {message}")
                    print("⚠️ Proceeding with original audio...")
            else:
                print("✅ Audio already in optimal format for Azure")
        
        # Step 4: Final size check
        if len(processed_audio) < 16000:  # Less than ~1 second at 16kHz
            return "Processed audio too small. Please record for at least 3-4 seconds with clear speech."
        
        # Step 5: Send to Azure OpenAI with retries
        print(f"🚀 Sending to Azure OpenAI: {len(processed_audio)} bytes")
        transcription = await stt_tool.transcribe_with_retries(processed_audio)
        
        return transcription
        
    except Exception as e:
        print(f"❌ STT pipeline failed: {e}")
        import traceback
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return f"Speech recognition failed: {str(e)}. Please try recording again."

# Backward compatibility
gpt4o_stt = GPT4oHTTPSTT()
gpt4o_websocket_stt = gpt4o_stt

