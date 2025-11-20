import google.generativeai as genai
import base64
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PIL import Image
import io
import time
import logging
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration
CONFIG = {
    'capture_interval': 20,  # seconds between frames - easily configurable
    'test_mode': False,  # Set to True to skip LLM calls
    'max_retries': 3,
    'retry_delay': 1,
}

# Global session state for real-time sharing with thread safety
class SessionState:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = False
        self.processed_count = 0
        self.latest_result = 'No results yet'
        self.last_update = None
        self.test_mode = False
        self.frame_number = 0
    
    def update_result(self, result):
        with self._lock:
            self.processed_count += 1
            self.frame_number += 1
            self.latest_result = result
            self.last_update = time.time()
            logger.info(f"Frame {self.frame_number} processed: {result}")
    
    def reset(self):
        with self._lock:
            self.active = False
            self.processed_count = 0
            self.frame_number = 0
            self.latest_result = 'Session ended'
            self.last_update = time.time()

session_state = SessionState()

class GeminiVisionService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def process_image(self, image_data, frame_number, test_mode=False):
        """Process image with Gemini Vision using current API"""
        try:
            if test_mode:
                # Test mode - generate predictable responses
                return self._generate_test_response(frame_number)
            
            if not self.api_key:
                return "❌ Error: Gemini API key not configured"

            # Convert to PIL Image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data

            # Use the current Gemini model
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = """
            Carefully analyze this image and focus **only on the Question** displayed on the screen. 
            Ignore any background text, images, decorations, or other irrelevant content.

            Instructions:
            1. If a question number is visible, put only the question number in the Q field.
            2. If no question number is visible, provide a very short summary of the question (2-5 words) in the Q field.
            3. Provide only the final answer in the A field (as per the MCQ answer).

            If no MCQ question is visible:
            - Simply say "No question detected" in Q and "N/A" in A.

            Format your response exactly as:
            Q: [question number or short summary or "No question detected"]
            A: [final answer or "N/A"]
            """

            # Prepare the image for the API
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data if isinstance(image_data, bytes) else self.pil_to_bytes(image)
            }
            
            # Retry mechanism for API calls
            for attempt in range(CONFIG['max_retries']):
                try:
                    response = model.generate_content([prompt, image_part])
                    
                    if response.text:
                        return response.text.strip()
                    else:
                        return f"Q: Frame {frame_number}\nA: No response from API"
                        
                except Exception as api_error:
                    if attempt < CONFIG['max_retries'] - 1:
                        logger.warning(f"API call failed, retrying... ({attempt + 1}/{CONFIG['max_retries']})")
                        time.sleep(CONFIG['retry_delay'])
                    else:
                        raise api_error
                        
            return f"Q: Frame {frame_number}\nA: API Error after retries"
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Q: Frame {frame_number}\nA: Processing error: {str(e)[:50]}..."

    def _generate_test_response(self, frame_number):
        """Generate test responses for testing without LLM calls"""
        test_responses = [
            "Q: Question 1\nA: Option C",
            "Q: Question 2\nA: Option B", 
            "Q: Physics Problem\nA: 9.8 m/s²",
            "Q: Math Equation\nA: x = 15",
            "Q: No question detected\nA: N/A"
        ]
        # Cycle through test responses based on frame number
        response_index = frame_number % len(test_responses)
        return test_responses[response_index]

    def pil_to_bytes(self, image):
        """Convert PIL Image to bytes"""
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        return buf.getvalue()

# Service instance
gemini_service = GeminiVisionService()

def home_view(request):
    """Camera UI - for taking pictures"""
    return render(request, 'vision_app/home.html', {
        'test_mode': CONFIG['test_mode'],
        'capture_interval': CONFIG['capture_interval']
    })

def display_view(request):
    """Display UI - for showing results"""
    return render(request, 'vision_app/display.html')

@csrf_exempt
def start_session(request):
    """Start processing session"""
    global session_state
    
    try:
        data = json.loads(request.body) if request.body else {}
        test_mode = data.get('test_mode', CONFIG['test_mode'])
        
        session_state.active = True
        session_state.processed_count = 0
        session_state.frame_number = 0
        session_state.test_mode = test_mode
        session_state.latest_result = 'Session started - waiting for first image...'
        session_state.last_update = time.time()
        
        logger.info(f"Session started - Test mode: {test_mode}")
        
        return JsonResponse({
            'status': 'success', 
            'message': 'Session started successfully',
            'test_mode': test_mode
        })
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to start session: {str(e)}'
        }, status=500)

@csrf_exempt
def end_session(request):
    """End processing session"""
    global session_state
    
    try:
        count = session_state.processed_count
        frame_number = session_state.frame_number
        session_state.reset()
        
        logger.info(f"Session ended. Processed {count} frames out of {frame_number} captured.")
        
        return JsonResponse({
            'status': 'success', 
            'processed_count': count,
            'total_frames': frame_number
        })
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to end session: {str(e)}'
        }, status=500)

@csrf_exempt
def upload_frame(request):
    """Process a single frame - robust error handling"""
    global session_state
    
    if not session_state.active:
        return JsonResponse({
            'status': 'error', 
            'message': 'No active session. Please start session first.'
        })

    try:
        # Get image data with robust error handling
        image_data = None
        try:
            if 'image' in request.FILES:
                image_data = request.FILES['image'].read()
            else:
                data = json.loads(request.body) if request.body else {}
                image_b64 = data.get('image', '')
                if ',' in image_b64:
                    image_b64 = image_b64.split(',')[1]
                image_data = base64.b64decode(image_b64)
        except Exception as e:
            logger.warning(f"Failed to extract image data: {str(e)}")
            # Create a dummy image for session continuity
            if session_state.test_mode:
                # Use test mode response
                result = gemini_service._generate_test_response(session_state.frame_number + 1)
                session_state.update_result(result)
                return JsonResponse({
                    'status': 'success',
                    'result': result,
                    'processed_count': session_state.processed_count,
                    'frame_number': session_state.frame_number
                })
            else:
                # Return error but keep session running
                result = f"Q: Frame {session_state.frame_number + 1}\nA: Image capture issue"
                session_state.update_result(result)
                return JsonResponse({
                    'status': 'success',  # Still return success to keep session running
                    'result': result,
                    'processed_count': session_state.processed_count,
                    'frame_number': session_state.frame_number
                })

        # Process with Gemini (or test mode)
        result = gemini_service.process_image(
            image_data, 
            session_state.frame_number + 1,
            session_state.test_mode
        )
        
        # Update global state for display
        session_state.update_result(result)
        
        return JsonResponse({
            'status': 'success',
            'result': result,
            'processed_count': session_state.processed_count,
            'frame_number': session_state.frame_number
        })
        
    except Exception as e:
        # Critical: Never break the session due to frame processing errors
        error_msg = f"Q: Frame {session_state.frame_number + 1}\nA: System error (session continues)"
        logger.error(f"Frame processing error (non-fatal): {str(e)}")
        
        # Still update state but mark as error
        session_state.update_result(error_msg)
        
        return JsonResponse({
            'status': 'success',  # Return success to keep session running
            'result': error_msg,
            'processed_count': session_state.processed_count,
            'frame_number': session_state.frame_number
        })

@csrf_exempt
def get_latest_result(request):
    """Get the latest processing result for display"""
    global session_state
    return JsonResponse({
        'active': session_state.active,
        'latest_result': session_state.latest_result,
        'processed_count': session_state.processed_count,
        'frame_number': session_state.frame_number,
        'last_update': session_state.last_update
    })

@csrf_exempt
def session_status(request):
    """Get current session status"""
    global session_state
    return JsonResponse({
        'active': session_state.active,
        'processed_count': session_state.processed_count,
        'frame_number': session_state.frame_number,
        'test_mode': session_state.test_mode
    })

@csrf_exempt
def update_config(request):
    """Update configuration (like capture interval and quality)"""
    global CONFIG
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            if 'capture_interval' in data:
                CONFIG['capture_interval'] = max(2, int(data['capture_interval']))
            if 'test_mode' in data:
                CONFIG['test_mode'] = bool(data['test_mode'])
            if 'quality' in data:
                CONFIG['quality'] = float(data['quality'])
            
            return JsonResponse({
                'status': 'success',
                'config': CONFIG
            })
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
    
    return JsonResponse({
        'status': 'success',
        'config': CONFIG
    })