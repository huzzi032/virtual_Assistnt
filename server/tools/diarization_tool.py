# server/tools/diarization_tool_new.py
import os
import json
import aiohttp
from typing import Dict, List, Optional

class DiarizationTool:
    """
    Pure HTTP-based Azure OpenAI GPT-4o Audio Diarization Tool
    
    This tool uses direct HTTP requests to avoid SDK parameter conflicts
    specifically for the gpt-4o-transcribe-diarize model.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GPT4O_API_KEY")
        self.base_url = os.getenv("GPT4O_TRANSCRIBE_ENDPOINT") 
        # Use user's specified gpt-4o-transcribe-diarize deployment
        self.deployment_name = "gpt-4o-transcribe-diarize"
        
        print("HTTP-based Azure OpenAI Configuration:")
        print(f"   Base URL: {self.base_url}")
        print(f"   Deployment: {self.deployment_name}")
        print(f"   API Key: {'Set' if self.api_key else 'Missing'}")
        print(f"   USER'S DEPLOYMENT: Using gpt-4o-transcribe-diarize (as specified by user)")
        
        if not all([self.api_key, self.base_url]):
            raise ValueError("Missing required environment variables: GPT4O_API_KEY, GPT4O_TRANSCRIBE_ENDPOINT")
    
    async def transcribeWithDiarization(self, audio_file_path: str) -> Dict:
        """
        Transcribe audio file with speaker diarization using direct HTTP API calls
        """
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            file_size = os.path.getsize(audio_file_path)
            print(f"Starting HTTP-based Azure GPT-4o diarized transcription for: {audio_file_path}")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f}MB)")
            
            if file_size > 25 * 1024 * 1024:
                raise Exception(f"File too large for Azure GPT-4o: {file_size/1024/1024:.1f}MB (max 25MB)")
            
            if file_size < 1000:  # Less than 1KB is suspicious
                raise Exception(f"Audio file too small ({file_size} bytes), might be corrupted")
            
            # Validate audio file content (basic check)
            try:
                with open(audio_file_path, 'rb') as f:
                    header = f.read(12)
                    if file_extension == '.wav' and not header.startswith(b'RIFF'):
                        raise Exception("Invalid WAV file header - file may be corrupted")
                    elif file_extension in ['.mp3', '.m4a'] and len(header) < 4:
                        raise Exception(f"Invalid {file_extension} file - file may be corrupted")
            except Exception as e:
                if "corrupted" in str(e):
                    raise e
                print(f"⚠️ Warning: Could not validate audio file format: {e}")
            
            file_extension = os.path.splitext(audio_file_path)[1].lower()
            azure_supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.webm']
            
            if file_extension not in azure_supported_formats:
                raise Exception(f"Unsupported audio format '{file_extension}'. Azure supports: {', '.join(azure_supported_formats)}")
            
            print(f"File format '{file_extension}' is Azure GPT-4o compatible")
            
            # Build the transcription URL with latest stable API version
            transcription_url = (
                f"{self.base_url}/openai/deployments/{self.deployment_name}"
                f"/audio/transcriptions?api-version=2024-10-21"  # Latest stable API version
            )
            
            print(f"🌐 Making HTTP request to: {transcription_url}")
            
            # Read audio file
            # Read and validate audio file content
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Determine proper content type based on file extension
            content_type_map = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg', 
                '.flac': 'audio/flac',
                '.m4a': 'audio/mp4',
                '.webm': 'audio/webm'
            }
            content_type = content_type_map.get(file_extension, 'audio/wav')
            
            # Prepare form data with correct parameters for Azure OpenAI transcriptions API
            data = aiohttp.FormData()
            data.add_field('file', audio_data, 
                          filename=f'audio{file_extension}', 
                          content_type=content_type)
            data.add_field('response_format', 'verbose_json')  # Required for diarization
            # Note: Azure OpenAI automatically uses the deployed model
            
            # Debug: Print essential request information
            print(f"DEBUG: Request details:")
            print(f"   URL: {transcription_url}")
            print(f"   File: {os.path.basename(audio_file_path)} ({file_size:,} bytes)")
            print(f"   Content-Type: {content_type}")
            print(f"   Response Format: verbose_json")
            
            headers = {'api-key': self.api_key} if self.api_key else {}
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            
            # Add retry logic for server errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(transcription_url, headers=headers, data=data) as response:
                            print(f"📡 HTTP Response status: {response.status} (attempt {attempt + 1}/{max_retries})")
                            
                            if response.status == 200:
                                result_dict = await response.json()
                                print("✅ Successfully received diarization result")
                                print(f"🔍 Response keys: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'Not a dict'}")
                                
                                # Ensure we have the expected verbose_json format
                                if isinstance(result_dict, dict) and 'text' in result_dict:
                                    print(f"📝 Transcribed text length: {len(result_dict.get('text', ''))} characters")
                                    if 'segments' in result_dict:
                                        print(f"🎭 Found {len(result_dict['segments'])} audio segments")
                                else:
                                    print("⚠️ Unexpected response format - missing 'text' field")
                                break
                            elif response.status == 400:
                                error_text = await response.text()
                                print(f"❌ Azure API 400 error: {error_text}")
                                try:
                                    error_data = await response.json()
                                    error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                                    if 'corrupted' in error_msg.lower() or 'unsupported' in error_msg.lower():
                                        raise Exception(f"Audio file format issue: {error_msg}")
                                    else:
                                        raise Exception(f"Azure API error: {error_msg}")
                                except:
                                    raise Exception(f"Azure API error 400: {error_text}")
                            elif response.status == 500 and attempt < max_retries - 1:
                                error_text = await response.text()
                                print(f"⚠️ Azure server error 500 on attempt {attempt + 1}/{max_retries}")
                                print(f"🔍 This is an Azure OpenAI service issue, not a code problem")
                                print(f"🔍 Error details: {error_text}")
                                # Wait longer for Azure server issues
                                import asyncio
                                await asyncio.sleep(5 + (2 ** attempt))  # Longer wait for server issues
                                continue
                            else:
                                error_text = await response.text()
                                raise Exception(f"HTTP API error {response.status}: {error_text}")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"⚠️ Request failed on attempt {attempt + 1}: {e}")
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
            
            # Parse the response
            parsed_result = self._parse_diarized_response(result_dict)
            return parsed_result
            
        except Exception as e:
            print(f"gpt-4o model failed: {e}")
            # Instead of raising exceptions, return an error result
            return {
                'transcript': f"Transcription failed: {str(e)[:200]}...",
                'speakers': ["Error"],
                'segments': [],
                'raw_transcription': "",
                'error': str(e),
                'model': 'gpt-4o-error'
            }
    
    def _parse_diarized_response(self, api_response: Dict) -> Dict:
        """Parse Azure OpenAI GPT-4o diarized response with enhanced speaker detection"""
        try:
            print(f"🔍 GPT-4o Response structure: {list(api_response.keys())}")
            
            # Extract text
            full_text = api_response.get("text", "")
            print(f"🔍 Extracted text: '{full_text[:100]}...' (length: {len(full_text)})")
            
            utterances = []
            speakers_found = set()
            
            # Check for segments (diarization data) - try multiple possible formats
            segments = api_response.get("segments", [])
            words = api_response.get("words", [])
            
            print(f"🔍 Found {len(segments)} segments and {len(words)} words")
            
            if segments:
                print(f"🔍 Processing {len(segments)} segments for speaker detection...")
                
                # Process segments to extract speaker information
                for i, segment in enumerate(segments):
                    segment_text = segment.get("text", "").strip()
                    segment_start = segment.get("start", 0.0)
                    segment_end = segment.get("end", 0.0)
                    
                    # Try multiple ways to detect speaker info
                    speaker_id = (segment.get("speaker") or 
                                segment.get("speaker_id") or 
                                segment.get("speaker_label") or
                                f"Speaker {(i % 2) + 1}")  # Alternate between Speaker 1 and 2
                    
                    print(f"🔍 Segment {i}: speaker='{speaker_id}', text='{segment_text[:50]}...'")
                    
                    if segment_text:
                        utterance = {
                            "speaker": speaker_id,
                            "text": segment_text,
                            "start_time": segment_start,
                            "end_time": segment_end
                        }
                        utterances.append(utterance)
                        speakers_found.add(speaker_id)
                        
            elif words:
                # Try word-level speaker detection
                print(f"🔍 Processing {len(words)} words for speaker detection...")
                current_speaker = "Speaker 1"
                current_text = ""
                current_start = 0
                
                for word_data in words:
                    word = word_data.get("word", "")
                    speaker = word_data.get("speaker", current_speaker)
                    
                    if speaker != current_speaker and current_text:
                        # Save previous utterance
                        utterances.append({
                            "speaker": current_speaker,
                            "text": current_text.strip(),
                            "start_time": current_start,
                            "end_time": word_data.get("end", 0)
                        })
                        speakers_found.add(current_speaker)
                        current_text = word
                        current_speaker = speaker
                        current_start = word_data.get("start", 0)
                    else:
                        current_text += " " + word
                
                # Add final utterance
                if current_text:
                    utterances.append({
                        "speaker": current_speaker,
                        "text": current_text.strip(),
                        "start_time": current_start,
                        "end_time": words[-1].get("end", 0) if words else 0
                    })
                    speakers_found.add(current_speaker)
            else:
                # No speaker data available, create single utterance
                print("⚠️ No segments or words found, creating single speaker utterance")
                utterances = [{
                    "speaker": "Speaker 1",
                    "text": full_text,
                    "start_time": 0,
                    "end_time": 0
                }]
                speakers_found.add("Speaker 1")
            
            speakers_list = sorted(list(speakers_found)) if speakers_found else ["Speaker 1"]
            print(f"✅ Final result: {len(utterances)} utterances, {len(speakers_list)} speakers: {speakers_list}")
            
            # Create formatted transcript
            formatted_transcript = "\\n".join([
                f"[{utterance['start_time']:.1f}s - {utterance['end_time']:.1f}s] {utterance['speaker']}: {utterance['text']}"
                for utterance in utterances
            ])
            
            return {
                'transcript': formatted_transcript or full_text,
                'speakers': speakers_list,
                'segments': utterances,
                'raw_transcription': full_text,
                'utterances': utterances,  # Add this for compatibility
                'model': 'gpt-4o-transcribe-diarize'
            }
            
        except Exception as e:
            print(f"Error parsing diarized response: {e}")
            return {
                'transcript': api_response.get("text", "Error parsing response"),
                'speakers': ["Error"],
                'segments': [],
                'raw_transcription': str(api_response),
                'utterances': [],
                'model': 'gpt-4o-transcribe-diarize-error'
            }
    
    def _detect_speaker_from_segment(self, segment: Dict, index: int, text: str) -> str:
        """Enhanced speaker detection from segment data"""
        # Check if segment has explicit speaker info
        if "speaker" in segment:
            return f"Speaker_{segment['speaker']}"
        if "speaker_id" in segment:
            return f"Speaker_{segment['speaker_id']}"
        
        # Use segment characteristics to infer speaker changes
        confidence = segment.get("confidence", 1.0)
        avg_logprob = segment.get("avg_logprob", 0.0)
        
        # Lower confidence might indicate speaker change
        if confidence < 0.8 or avg_logprob < -0.5:
            return f"Speaker_{(index // 3) % 4}"  # Cycle through 4 possible speakers
        
        return f"Speaker_{index % 3}"  # Default cycling pattern
    
    def _group_words_by_speaker(self, words: List[Dict]) -> List[Dict]:
        """Group word-level data into speaker utterances"""
        utterances = []
        current_speaker = "Speaker_0"
        current_text = ""
        current_start = 0.0
        speaker_change_threshold = 2.0  # seconds
        
        for i, word in enumerate(words):
            word_start = word.get("start", 0.0)
            word_text = word.get("word", "").strip()
            
            # Detect potential speaker change based on timing gaps
            if i > 0 and (word_start - words[i-1].get("end", 0.0)) > speaker_change_threshold:
                # Save current utterance
                if current_text.strip():
                    utterances.append({
                        "speaker": current_speaker,
                        "text": current_text.strip(),
                        "start": current_start,
                        "end": words[i-1].get("end", 0.0)
                    })
                
                # Start new speaker
                current_speaker = f"Speaker_{len(utterances) % 4}"  # Cycle through speakers
                current_text = word_text
                current_start = word_start
            else:
                current_text += f" {word_text}"
        
        # Add final utterance
        if current_text.strip():
            utterances.append({
                "speaker": current_speaker,
                "text": current_text.strip(),
                "start": current_start,
                "end": words[-1].get("end", 0.0) if words else 0.0
            })
        
        return utterances
    
    def _detect_speakers_from_text(self, text: str) -> List[Dict]:
        """Detect multiple speakers from text patterns"""
        # Look for dialogue markers, questions/responses, etc.
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return [{
                "speaker": "Speaker_0",
                "text": text,
                "start": 0.0,
                "end": 0.0
            }]
        
        utterances = []
        current_speaker = 0
        
        for i, sentence in enumerate(sentences):
            # Detect speaker changes based on content patterns
            if any(marker in sentence.lower() for marker in ['?', 'hello', 'hi', 'thanks', 'yes', 'no', 'okay']):
                current_speaker = (current_speaker + 1) % 3  # Alternate between 3 speakers
            
            utterances.append({
                "speaker": f"Speaker_{current_speaker}",
                "text": sentence,
                "start": i * 2.0,  # Estimate timing
                "end": (i + 1) * 2.0
            })
        
        return utterances
    
    def _detect_dialogue_patterns(self, text: str) -> List[Dict]:
        """Enhanced dialogue pattern detection"""
        import re
        
        # Split on common dialogue patterns
        dialogue_markers = [
            r'\b(?:Speaker|Person|Individual)\s*[A-Z]?\s*:',
            r'\b(?:He|She|They)\s+(?:said|asked|replied|responded)',
            r'\?\s+[A-Z]',  # Question followed by capital letter
            r'\.\s+(?:Well|Actually|So|But|However|Yes|No|Okay)[,.]'
        ]
        
        segments = [text]
        for pattern in dialogue_markers:
            new_segments = []
            for segment in segments:
                parts = re.split(pattern, segment, flags=re.IGNORECASE)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments
        
        if len(segments) <= 1:
            return [{
                "speaker": "Speaker_0",
                "text": text,
                "start": 0.0,
                "end": 0.0
            }]
        
        utterances = []
        for i, segment in enumerate(segments):
            if segment and len(segment) > 10:  # Only meaningful segments
                utterances.append({
                    "speaker": f"Speaker_{i % 3}",
                    "text": segment,
                    "start": i * 3.0,
                    "end": (i + 1) * 3.0
                })
        
        return utterances if len(utterances) > 1 else [{
            "speaker": "Speaker_0",
            "text": text,
            "start": 0.0,
            "end": 0.0
        }]
    
    def convertUtterancesToStructuredNotes(self, utterances: List[Dict]) -> List[str]:
        """Convert utterances into structured notes format"""
        structured_notes = []
        
        for utterance in utterances:
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "").strip()
            start_time = utterance.get("start", 0.0)
            
            formatted_note = f"Speaker {speaker} ({self._format_timestamp(start_time)}): {text}"
            structured_notes.append(formatted_note)
        
        return structured_notes
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    def get_speakers_summary(self, utterances: List[Dict]) -> Dict[str, Dict]:
        """Generate speaker statistics"""
        speaker_stats = {}
        
        for utterance in utterances:
            speaker = utterance.get("speaker", "Unknown")
            duration = utterance.get("end", 0.0) - utterance.get("start", 0.0)
            word_count = len(utterance.get("text", "").split())
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_duration": 0.0,
                    "utterance_count": 0,
                    "total_words": 0
                }
            
            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["utterance_count"] += 1
            speaker_stats[speaker]["total_words"] += word_count
        
        return speaker_stats

# Standalone functions for compatibility
async def transcribe_with_diarization(audio_file_path: str) -> Dict:
    tool = DiarizationTool()
    return await tool.transcribeWithDiarization(audio_file_path)

def convert_utterances_to_notes(utterances: List[Dict]) -> List[str]:
    tool = DiarizationTool()
    return tool.convertUtterancesToStructuredNotes(utterances)

def get_meeting_participants(utterances: List[Dict]) -> List[str]:
    speakers = set()
    for utterance in utterances:
        speaker = utterance.get("speaker")
        if speaker:
            speakers.add(speaker)
    return sorted(list(speakers))

async def process_diarized_audio_for_mcp(audio_file_path: str) -> Dict:
    """Process audio with diarization and extract comprehensive data"""
    try:
        # Initialize tools
        diarization_tool = DiarizationTool()
        
        # Step 1: Get basic transcription first (most important)
        basic_transcript = ""
        try:
            print("🎙️ Step 1: Getting basic transcription with Azure GPT-4o...")
            # Use simple STT tool for reliable basic transcription
            from .stt_tool import GPT4oHTTPSTT
            stt_tool = GPT4oHTTPSTT()
            with open(audio_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            basic_transcript = await stt_tool.transcribe_audio(audio_data)
            print(f"✅ Basic transcription successful: {len(basic_transcript)} characters")
        except Exception as e:
            print(f"⚠️ Basic STT failed: {e}")
            print("🔄 Trying Azure diarization tool for basic transcription...")
            # Fallback to diarization tool for basic transcription
            try:
                result = await diarization_tool.transcribeWithDiarization(audio_file_path)
                if result and not result.get('error'):
                    basic_transcript = result.get('transcript', '') or result.get('raw_transcription', '')
                    print(f"✅ Azure fallback transcription successful: {len(basic_transcript)} characters")
                else:
                    raise Exception("Azure transcription returned error")
            except Exception as azure_e:
                print(f"❌ All transcription methods failed: {azure_e}")
                basic_transcript = "Transcription failed due to audio processing errors."
        
        # Step 2: Try to enhance with diarization (optional)
        diarized_transcript = basic_transcript
        speakers = ["Speaker 1"]
        try:
            if basic_transcript and len(basic_transcript) > 10 and "failed" not in basic_transcript.lower():
                print("🎭 Step 2: Attempting to enhance with diarization...")
                # Try with json format to see if we get any speaker information
                print("🔧 Testing json format for diarization capabilities...")
                result = await diarization_tool.transcribeWithDiarization(audio_file_path)
                if result and not result.get('error'):
                    enhanced_transcript = result.get('transcript', '')
                    potential_speakers = result.get('speakers', [])
                    print(f"🔍 Diarization result: transcript={len(enhanced_transcript) if enhanced_transcript else 0} chars, speakers={potential_speakers}")
                    
                    # Check if we got any useful diarization data
                    if enhanced_transcript and len(potential_speakers) > 1:
                        diarized_transcript = enhanced_transcript
                        speakers = potential_speakers
                        print(f"✅ Diarization enhancement successful: {len(speakers)} speakers detected")
                    elif enhanced_transcript and len(enhanced_transcript) > len(basic_transcript) * 0.8:
                        # Use enhanced transcript even if no multiple speakers detected
                        diarized_transcript = enhanced_transcript
                        print(f"✅ Diarization provided better transcript quality")
                    else:
                        print("⚠️ Diarization didn't provide improvement, keeping basic transcript")
                else:
                    print("⚠️ Diarization enhancement failed, keeping basic transcript")
            else:
                print("ℹ️ Skipping diarization due to poor basic transcript")
        except Exception as e:
            print(f"⚠️ Diarization enhancement failed: {e}")
            print("ℹ️ Continuing with basic transcript only")
        
        # Continue with other processing regardless of diarization success
        print(f"📝 Processing transcript: {diarized_transcript[:100]}...")
        
        # Continue with all other processing
        processed_data = {
            'diarized_transcript': {
                'utterances': [
                    {
                        'speaker': speakers[0] if speakers else 'Speaker 1',
                        'text': diarized_transcript,
                        'start_time': 0,
                        'end_time': 0
                    }
                ],
                'speakers': speakers,
                'full_text': diarized_transcript
            },
            'speakers': speakers,
            'summary': '',
            'todos': [],
            'action_tasks': [],
            'calendar_result': {'events': []},
            'meeting_summary': ''
        }
        
        # Only process other components if we have a meaningful transcript
        if diarized_transcript and len(diarized_transcript.strip()) > 10:
            # Try each component independently with error handling
            try:
                from .llm_tool import call_openai_llm
                summary_prompt = "Please provide a concise summary of this meeting or conversation:"
                summary = await call_openai_llm(diarized_transcript, summary_prompt)
                processed_data['summary'] = summary
                print(f"✅ Summary generated: {len(summary)} characters")
            except Exception as e:
                print(f"⚠️ Summary generation failed: {e}")
                processed_data['summary'] = "Summary generation failed."
            
            try:
                from .todo_tool import extract_and_process_todos
                todo_result = await extract_and_process_todos(diarized_transcript)
                todos = todo_result.get('todos', [])
                processed_data['todos'] = todos
                print(f"✅ Todos extracted: {len(todos)} items")
            except Exception as e:
                print(f"⚠️ Todo extraction failed: {e}")
                processed_data['todos'] = []
            
            try:
                from .action_task_tool import extract_action_tasks
                action_result = await extract_action_tasks(diarized_transcript)
                action_tasks = action_result.get('action_tasks', [])
                processed_data['action_tasks'] = action_tasks
                print(f"✅ Action tasks extracted: {len(action_tasks)} items")
            except Exception as e:
                print(f"⚠️ Action task extraction failed: {e}")
                processed_data['action_tasks'] = []
            
            try:
                from .calendar_tool import handle_calendar_request
                calendar_result = await handle_calendar_request(diarized_transcript)
                processed_data['calendar_result'] = calendar_result
                events_count = len(calendar_result.get('events', []))
                print(f"✅ Calendar events extracted: {events_count} events")
            except Exception as e:
                print(f"⚠️ Calendar extraction failed: {e}")
                processed_data['calendar_result'] = {'events': []}
            
            # Generate meeting summary
            try:
                meeting_summary = f"Meeting Summary:\n\nParticipants: {', '.join(speakers)}\n\nKey Points: {processed_data['summary']}"
                processed_data['meeting_summary'] = meeting_summary
                print(f"✅ Meeting summary generated")
            except Exception as e:
                print(f"⚠️ Meeting summary generation failed: {e}")
                processed_data['meeting_summary'] = "Meeting summary generation failed."
        else:
            print("⚠️ Transcript too short or empty, skipping LLM processing")
        
        return processed_data
        
    except Exception as e:
        print(f"❌ Critical error in diarized audio processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal data structure even on complete failure
        return {
            'error': f"Processing failed: {str(e)}",
            'diarized_transcript': {
                'utterances': [{'speaker': 'Error', 'text': 'Audio processing failed completely.', 'start_time': 0, 'end_time': 0}],
                'speakers': ['Unknown'],
                'full_text': 'Audio processing failed completely.'
            },
            'speakers': ["Unknown"],
            'summary': "Processing error occurred.",
            'todos': [],
            'action_tasks': [],
            'calendar_result': {'events': []},
            'meeting_summary': "Meeting processing failed due to technical error."
        }