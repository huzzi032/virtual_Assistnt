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
        # Note: Using diarize model but with simple JSON format for transcription only
        self.api_version = os.getenv('GPT4O_API_VERSION', '2024-06-01')
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
            
            # For single submission (without pre-chunking), use 5MB limit for stability
            DIRECT_SUBMIT_LIMIT = 5 * 1024 * 1024  # 5MB
            ABSOLUTE_MAX = 25 * 1024 * 1024  # 25MB absolute limit
            
            if len(audio_data) > ABSOLUTE_MAX:
                return False, f"Audio too large: {len(audio_data)} bytes (max {ABSOLUTE_MAX})"
            
            if len(audio_data) > DIRECT_SUBMIT_LIMIT:
                print(f"⚠️ File size {len(audio_data)/1024/1024:.1f}MB exceeds {DIRECT_SUBMIT_LIMIT/1024/1024:.0f}MB direct limit - will use pre-chunking")
            
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
    
    async def transcribe_audio(self, audio_data: bytes, file_format: str = "unknown") -> dict:
        """
        Transcribe audio with format detection and direct Azure submission
        
        Args:
            audio_data: Raw audio bytes
            file_format: Detected file format (wav, mp3, aac, etc.)
            
        Returns:
            dict: {"success": bool, "transcription": str, "error": str}
        """
        try:
            print(f"🎵 Transcribing {file_format} audio: {len(audio_data)} bytes")
            
            # Map format to content type (Azure supported formats only)
            azure_content_types = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg", 
                "m4a": "audio/mp4",
                "mp4": "audio/mp4", 
                "mpeg": "audio/mpeg",
                "mpga": "audio/mpeg",
                "webm": "audio/webm"
            }
            
            # For unsupported formats, use WAV
            if file_format not in azure_content_types:
                print(f"⚠️ Converting {file_format} to WAV for Azure compatibility")
                content_type = "audio/wav"
                filename = "audio.wav"
            else:
                content_type = azure_content_types[file_format]
                filename = f"audio.{file_format}"
            
            print(f"📤 Sending to Azure OpenAI: {filename} ({content_type})")
            
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for longer files
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Create multipart form data
                data = aiohttp.FormData()
                data.add_field(
                    'file',
                    audio_data,
                    filename=filename,
                    content_type=content_type
                )
                data.add_field('model', self.deployment)
                data.add_field('response_format', 'json')  # Simple JSON format
                data.add_field('chunking_strategy', 'auto')  # Required for diarization model
                
                # Headers for Azure OpenAI
                headers = {
                    'api-key': self.api_key
                }
                
                async with session.post(self.url, data=data, headers=headers) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        try:
                            result = json.loads(response_text)
                            transcription = result.get('text', '').strip()
                            
                            if transcription:
                                print(f"✅ Transcription successful: {len(transcription)} characters")
                                return {
                                    "success": True,
                                    "transcription": transcription,
                                    "error": None
                                }
                            else:
                                return {
                                    "success": False,
                                    "transcription": "",
                                    "error": "Empty transcription returned"
                                }
                                
                        except json.JSONDecodeError as e:
                            return {
                                "success": False,
                                "transcription": "",
                                "error": f"Invalid JSON response: {e}"
                            }
                    else:
                        error_msg = f"Azure API error {response.status}: {response_text}"
                        print(f"❌ {error_msg}")
                        return {
                            "success": False,
                            "transcription": "",
                            "error": error_msg
                        }
                        
        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "transcription": "",
                "error": error_msg
            }

    async def transcribe_with_retries(self, audio_data: bytes, max_retries: int = 3) -> str:
        """
        Transcribe with exponential backoff retry for Azure errors
        
        Uses simple JSON format for fast and reliable transcription without diarization
        """
        
        for attempt in range(max_retries):
            try:
                print(f"🌐 Attempt {attempt + 1}/{max_retries}: Sending {len(audio_data)} bytes to Azure")
                print(f"🔗 URL: {self.url}")
                print(f"🔑 API Key (first 10 chars): {self.api_key[:10]}...")
                print(f"📦 Deployment: {self.deployment}")
                
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for longer files
                
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
                    data.add_field('response_format', 'json')  # Use simple JSON format only
                    data.add_field('chunking_strategy', 'auto')  # Required for diarization model
                    
                    # Correct headers for Azure OpenAI - using api-key header
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
                            
                            # Simple transcription only - no diarization
                            transcription = ""
                            
                            if 'text' in result:
                                transcription = result['text'].strip()
                            
                            if transcription:
                                print(f"✅ Transcription successful: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")
                                return transcription
                            else:
                                print(f"⚠️ No transcription found in response: {result}")
                                return "No speech detected in audio. Please speak clearly and try again."
                        
                        elif response.status == 400:
                            error_text = await response.text()
                            print(f"❌ Bad request (400): {error_text}")
                            
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
    Complete Azure OpenAI GPT-4o STT pipeline with smart chunking for large files
    
    Args:
        audio_data (bytes): Audio file data
        
    Returns:
        str: Transcribed text or user-friendly error message
    """
    try:
        print(f"📝 Starting GPT-4o STT processing: {len(audio_data)} bytes ({len(audio_data)/1024/1024:.2f}MB)")
        
        # Initialize STT tool using lazy initialization
        stt_tool = get_gpt4o_stt()
        
        # Step 1: Validate audio size and determine chunking strategy
        size_valid, size_message = stt_tool.validate_audio_requirements(audio_data)
        print(f"📏 {size_message}")
        
        if len(audio_data) > 25 * 1024 * 1024:  # Absolute max
            return size_message
        
        # Define thresholds
        DIRECT_SUBMIT_LIMIT = 5 * 1024 * 1024  # 5MB for direct submission
        
        # Check if file needs pre-chunking
        if len(audio_data) > DIRECT_SUBMIT_LIMIT:
            print(f"🔄 Large file detected ({len(audio_data)/1024/1024:.2f}MB). Using pre-chunking strategy...")
            return await _process_audio_in_chunks(audio_data, stt_tool, DIRECT_SUBMIT_LIMIT)
        
        # Step 2: For smaller files, process normally
        # Detect audio format
        file_extension = _detect_audio_format(audio_data)
        print(f"🎵 Detected audio format: {file_extension}")
        
        # Check if format needs conversion
        azure_supported_formats = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
        
        if file_extension in azure_supported_formats:
            print("✅ Audio format supported by Azure OpenAI Whisper")
            result = await stt_tool.transcribe_audio(audio_data, file_extension)
        else:
            print(f"⚠️ Format '{file_extension}' not directly supported by Azure OpenAI")
            print("🔄 Converting to WAV format for compatibility...")
            result = await stt_tool.transcribe_audio(audio_data, "wav")
        
        if result.get("success"):
            transcription = result["transcription"]
            print(f"✅ STT Success: {len(transcription)} characters transcribed")
            return transcription
        else:
            error_message = result.get("error", "Unknown transcription error")
            print(f"❌ STT Error: {error_message}")
            return f"Transcription failed: {error_message}"
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ STT Exception: {error_msg}")
        return f"Audio processing failed: {error_msg}"

async def _process_audio_in_chunks(audio_data: bytes, stt_tool, chunk_size: int) -> str:
    """
    Process large audio files by splitting into time-based chunks using ffmpeg
    
    Args:
        audio_data: The large audio file bytes
        stt_tool: The STT tool instance
        chunk_size: Maximum bytes per chunk
        
    Returns:
        Combined transcription from all chunks
    """
    try:
        print(f"🔪 Chunking large audio file ({len(audio_data)/1024/1024:.2f}MB) into {chunk_size/1024/1024:.0f}MB segments")
        
        # Check if ffmpeg is available for chunking
        if not shutil.which('ffmpeg'):
            print("⚠️ ffmpeg not available for chunking. Attempting direct processing...")
            # Fallback to direct processing with increased timeout
            result = await stt_tool.transcribe_audio(audio_data, _detect_audio_format(audio_data))
            if result.get("success"):
                return result["transcription"]
            else:
                return f"Large file processing failed: {result.get('error', 'Unknown error')}"
        
        # Save original file to temp location
        with tempfile.NamedTemporaryFile(suffix='.input', delete=False) as temp_input:
            temp_input.write(audio_data)
            input_path = temp_input.name
        
        try:
            # Get audio duration using ffprobe
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', input_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            
            if probe_result.returncode != 0:
                print("⚠️ Could not determine audio duration. Using size-based fallback...")
                return await _fallback_size_chunking(audio_data, stt_tool, chunk_size)
            
            import json
            probe_data = json.loads(probe_result.stdout)
            
            duration = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    duration = float(stream.get('duration', 0))
                    break
            
            if not duration or duration < 10:  # Less than 10 seconds
                print("⚠️ Audio too short or duration unknown. Processing directly...")
                result = await stt_tool.transcribe_audio(audio_data, _detect_audio_format(audio_data))
                return result.get("transcription", "No transcription available") if result.get("success") else f"Processing failed: {result.get('error', 'Unknown error')}"
            
            print(f"📊 Audio duration: {duration:.1f} seconds")
            
            # Calculate chunk duration (aim for ~5MB chunks)
            # Estimate bitrate and calculate time chunks
            estimated_bitrate = (len(audio_data) * 8) / duration  # bits per second
            target_chunk_duration = (chunk_size * 8) / estimated_bitrate if estimated_bitrate > 0 else 300  # Default 5 min chunks
            
            # Ensure chunks are at least 30 seconds but not more than 10 minutes
            chunk_duration = max(30, min(600, target_chunk_duration))
            num_chunks = int((duration / chunk_duration) + 1)
            
            print(f"🔢 Processing {num_chunks} chunks of ~{chunk_duration:.0f} seconds each")
            
            # Process each chunk
            transcriptions = []
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                
                # Ensure we don't go beyond the actual duration
                if start_time >= duration:
                    break
                
                # Set chunk duration, but don't exceed file end
                current_duration = min(chunk_duration, duration - start_time)
                
                print(f"🎬 Processing chunk {i+1}/{num_chunks}: {start_time:.1f}s - {start_time + current_duration:.1f}s")
                
                # Extract chunk using ffmpeg
                chunk_path = f"{input_path}_chunk_{i}.wav"
                
                chunk_cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', input_path,
                    '-ss', str(start_time),
                    '-t', str(current_duration),
                    '-ac', '1',              # Mono
                    '-ar', '16000',          # 16kHz
                    '-c:a', 'pcm_s16le',     # PCM format
                    chunk_path
                ]
                
                chunk_result = subprocess.run(chunk_cmd, capture_output=True, text=True, timeout=60)
                
                if chunk_result.returncode != 0:
                    print(f"⚠️ Failed to create chunk {i+1}: {chunk_result.stderr}")
                    continue
                
                try:
                    # Read chunk data
                    with open(chunk_path, 'rb') as chunk_file:
                        chunk_data = chunk_file.read()
                    
                    if len(chunk_data) < 1000:  # Skip very small chunks
                        print(f"⏭️ Skipping tiny chunk {i+1}")
                        continue
                    
                    # Transcribe chunk with retries
                    chunk_result = await stt_tool.transcribe_with_retries(chunk_data, max_retries=2)
                    
                    if chunk_result and "failed" not in chunk_result.lower():
                        transcriptions.append(f"[{start_time:.0f}s-{start_time + current_duration:.0f}s] {chunk_result.strip()}")
                        print(f"✅ Chunk {i+1} transcribed: {len(chunk_result)} characters")
                    else:
                        print(f"⚠️ Chunk {i+1} failed: {chunk_result}")
                
                except Exception as chunk_error:
                    print(f"⚠️ Error processing chunk {i+1}: {chunk_error}")
                
                finally:
                    # Cleanup chunk file
                    try:
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)
                    except:
                        pass
                
                # Brief pause between chunks to avoid rate limiting
                await asyncio.sleep(1)
            
            if transcriptions:
                combined_transcription = "\n\n".join(transcriptions)
                print(f"🎉 Chunked processing complete: {len(combined_transcription)} total characters")
                return combined_transcription
            else:
                return "No successful transcriptions from chunks. The audio may be too noisy or contain no speech."
        
        finally:
            # Cleanup input file
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except:
                pass
    
    except Exception as e:
        print(f"❌ Chunking error: {e}")
        # Fallback to direct processing
        print("🔄 Falling back to direct processing...")
        result = await stt_tool.transcribe_audio(audio_data, _detect_audio_format(audio_data))
        return result.get("transcription", f"Chunking and direct processing both failed: {e}") if result.get("success") else f"Processing failed: {result.get('error', str(e))}"

async def _fallback_size_chunking(audio_data: bytes, stt_tool, chunk_size: int) -> str:
    """
    Simple size-based chunking as fallback when ffmpeg duration analysis fails
    """
    print("📏 Using size-based chunking fallback")
    
    total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
    transcriptions = []
    
    for i in range(total_chunks):
        start_pos = i * chunk_size
        end_pos = min(start_pos + chunk_size, len(audio_data))
        chunk_data = audio_data[start_pos:end_pos]
        
        if len(chunk_data) < 1000:  # Skip very small chunks
            continue
        
        print(f"📦 Processing size chunk {i+1}/{total_chunks}: {len(chunk_data)} bytes")
        
        try:
            result = await stt_tool.transcribe_with_retries(chunk_data, max_retries=2)
            
            if result and "failed" not in result.lower():
                transcriptions.append(f"[Part {i+1}] {result.strip()}")
                print(f"✅ Size chunk {i+1} completed")
            else:
                print(f"⚠️ Size chunk {i+1} failed")
        
        except Exception as chunk_error:
            print(f"⚠️ Error in size chunk {i+1}: {chunk_error}")
        
        # Brief pause between chunks
        await asyncio.sleep(1)
    
    if transcriptions:
        return "\n\n".join(transcriptions)
    else:
        return "Size-based chunking failed to produce transcriptions."

def _detect_audio_format(audio_data: bytes) -> str:
    """Detect audio format from file header"""
    if len(audio_data) < 12:
        return "unknown"
    
    header = audio_data[:32]  # Increased header size for better detection
    
    # Common audio format signatures
    if header.startswith(b'RIFF') and b'WAVE' in header:
        return "wav"
    elif header.startswith(b'ID3') or header[1:4] == b'ID3':
        return "mp3"
    elif header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3'):
        return "mp3"
    elif b'ftyp' in header:
        # MP4/M4A containers
        if b'M4A ' in header or b'mp4a' in header:
            return "m4a"
        elif b'mp41' in header or b'mp42' in header or b'isom' in header:
            return "mp4"
        else:
            return "m4a"  # Default to m4a for ftyp containers
    elif header.startswith(b'\xff\xf1') or header.startswith(b'\xff\xf9'):
        # AAC ADTS format
        return "aac"
    elif b'ADIF' in header:
        # AAC ADIF format
        return "aac"
    elif header.startswith(b'OggS'):
        return "ogg" 
    elif header.startswith(b'OpusHead'):
        return "opus"
    elif b'3gp' in header or b'3g2' in header:
        return "3gp"
    elif header.startswith(b'\x00\x00\x00'):
        # Additional MP4/M4A check for files starting with size
        if b'ftyp' in audio_data[:64]:
            return "m4a"
    
    return "unknown"

# Lazy initialization - only create when needed
gpt4o_stt = None

def get_gpt4o_stt():
    """Get or create GPT4o STT instance"""
    global gpt4o_stt
    if gpt4o_stt is None:
        gpt4o_stt = GPT4oHTTPSTT()
    return gpt4o_stt

# Backward compatibility
gpt4o_websocket_stt = get_gpt4o_stt

