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

# Global session state for real-time sharing
session_state = {
    'active': False,
    'processed_count': 0,
    'latest_result': 'No results yet',
    'last_update': None
}

class GeminiVisionService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def process_image(self, image_data):
        """Process image with Gemini Vision using current API"""
        try:
            if not self.api_key:
                return "❌ Error: Gemini API key not configured"

            # Convert to PIL Image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data

            # Use the current Gemini model (gemini-1.5-flash or gemini-1.5-pro)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = """
            Analyze this image carefully:

            If there is a clear question, math problem, or text requiring an answer:
            1. If a question number is visible, put only the question number in the Q field.
            2. If no question number is visible, provide a very short summary of the question (2-5 words) in the Q field.
            3. Provide only the final answer in the A field (as per the MCQ answer).

            If no question is visible:
            - Simply say "No question detected" in Q and "N/A" in A.

            Format your response as:
            Q: [question number or short summary or "No question detected"]
            A: [final answer or "N/A"]
            """
            
            # Prepare the image for the API
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data if isinstance(image_data, bytes) else self.pil_to_bytes(image)
            }
            
            response = model.generate_content([prompt, image_part])
            
            if response.text:
                return response.text.strip()
            else:
                return "Q: Error\nA: Failed to process image"
                
        except Exception as e:
            return f"Q: Error\nA: {str(e)}"

    def pil_to_bytes(self, image):
        """Convert PIL Image to bytes"""
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        return buf.getvalue()

# Service instance
gemini_service = GeminiVisionService()

def home_view(request):
    """Camera UI - for taking pictures"""
    return render(request, 'vision_app/home.html')

def display_view(request):
    """Display UI - for showing results"""
    return render(request, 'vision_app/display.html')

@csrf_exempt
def start_session(request):
    """Start processing session"""
    global session_state
    session_state['active'] = True
    session_state['processed_count'] = 0
    session_state['latest_result'] = 'Session started - waiting for first image...'
    session_state['last_update'] = time.time()
    
    return JsonResponse({
        'status': 'success', 
        'message': 'Session started successfully'
    })

@csrf_exempt
def end_session(request):
    """End processing session"""
    global session_state
    count = session_state['processed_count']
    session_state['active'] = False
    session_state['latest_result'] = f'Session ended. Processed {count} frames.'
    session_state['last_update'] = time.time()
    
    return JsonResponse({
        'status': 'success', 
        'processed_count': count
    })

@csrf_exempt
def upload_frame(request):
    """Process a single frame"""
    global session_state
    
    if not session_state['active']:
        return JsonResponse({
            'status': 'error', 
            'message': 'No active session. Please start session first.'
        })

    try:
        # Get image data
        if 'image' in request.FILES:
            image_data = request.FILES['image'].read()
        else:
            data = json.loads(request.body)
            image_b64 = data.get('image', '').split(',')[1]
            image_data = base64.b64decode(image_b64)

        # Process with Gemini
        result = gemini_service.process_image(image_data)
        
        # Update global state for display
        session_state['latest_result'] = result
        session_state['processed_count'] += 1
        session_state['last_update'] = time.time()
        
        return JsonResponse({
            'status': 'success',
            'result': result,
            'processed_count': session_state['processed_count']
        })
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        session_state['latest_result'] = error_msg
        return JsonResponse({
            'status': 'error', 
            'message': str(e)
        }, status=500)

@csrf_exempt
def get_latest_result(request):
    """Get the latest processing result for display"""
    global session_state
    return JsonResponse({
        'active': session_state['active'],
        'latest_result': session_state['latest_result'],
        'processed_count': session_state['processed_count'],
        'last_update': session_state['last_update']
    })

@csrf_exempt
def session_status(request):
    """Get current session status"""
    global session_state
    return JsonResponse({
        'active': session_state['active'],
        'processed_count': session_state['processed_count']
    })