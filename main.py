import os
import sys
import shutil
from datetime import datetime
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                           QInputDialog, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from pynput import keyboard
import mss
import mss.tools
from threading import Thread
import time
import base64
from together import Together
from openai import OpenAI
from dotenv import load_dotenv
import markdown2
from PIL import Image
import io
import re
import requests
import html
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

class AnalysisWorker(QObject):
    """Worker class for handling analysis in a separate thread"""
    analysis_complete = pyqtSignal(str, list)  # content, citations
    analysis_error = pyqtSignal(str)
    analysis_progress = pyqtSignal(str)

    def __init__(self, together_client, perplexity_client):
        super().__init__()
        self.together_client = together_client
        self.perplexity_client = perplexity_client

    def analyze_screenshot(self, image_path):
        """Analyze screenshot in a separate thread"""
        try:
            # First, analyze with Llama Vision
            image_analysis = self.analyze_with_llama(image_path)
            
            # Then verify with Perplexity
            verification_result = self.verify_with_perplexity(image_analysis)
            
            # Extract content and citations from the response
            content = verification_result.choices[0].message.content
            citations = verification_result.citations if hasattr(verification_result, 'citations') else []
            
            # Convert markdown to HTML
            html_content = markdown2.markdown(content)
            
            # Create source cards if available
            sources_html = ""
            if citations:
                sources_html = """
                    <div style='margin-top: 30px; padding: 20px 0;'>
                        <h3 style='color: #2c3e50; font-size: 1.3em; margin-bottom: 15px;'>Sources</h3>
                        <div style='overflow-x: auto; white-space: nowrap; padding: 10px 0; margin: 0 -10px;'>
                """
                for i, citation in enumerate(citations, 1):
                    # Extract domain from URL
                    domain = citation.split('/')[2] if len(citation.split('/')) > 2 else citation
                    sources_html += f"""
                        <div style='
                            display: inline-block;
                            width: 300px;
                            margin: 0 10px;
                            padding: 15px;
                            background: white;
                            border: 1px solid #e0e0e0;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            vertical-align: top;
                            transition: transform 0.2s, box-shadow 0.2s;
                            cursor: pointer;
                            white-space: normal;
                        ' onmouseover='this.style.transform="translateY(-2px)";this.style.boxShadow="0 4px 8px rgba(0,0,0,0.15)"'
                           onmouseout='this.style.transform="none";this.style.boxShadow="0 2px 4px rgba(0,0,0,0.1)"'
                           onclick='window.open("{citation}", "_blank")'>
                            <div style='color: #1a73e8; font-weight: 500; margin-bottom: 8px;'>Source {i}</div>
                            <div style='color: #202124; font-size: 0.9em; margin-bottom: 10px;'>{domain}</div>
                            <div style='
                                color: #5f6368;
                                font-size: 0.8em;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                display: -webkit-box;
                                -webkit-line-clamp: 2;
                                -webkit-box-orient: vertical;
                            '>{citation}</div>
                        </div>
                    """
                sources_html += "</div></div>"
            
            # Wrap in styled container with modern design
            final_html = f"""
                <style>
                    * {{
                        box-sizing: border-box;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                        margin-top: 1.5em;
                        margin-bottom: 0.8em;
                    }}
                    p {{
                        color: #34495e;
                        line-height: 1.6;
                        margin-bottom: 1.2em;
                    }}
                    ul, ol {{
                        color: #34495e;
                        line-height: 1.6;
                        padding-left: 1.5em;
                    }}
                    li {{
                        margin-bottom: 0.5em;
                    }}
                    code {{
                        background: #f8f9fa;
                        padding: 0.2em 0.4em;
                        border-radius: 3px;
                        font-size: 0.9em;
                        color: #e83e8c;
                    }}
                    blockquote {{
                        border-left: 4px solid #1a73e8;
                        margin: 1.5em 0;
                        padding: 0.5em 1em;
                        background: #f8f9fa;
                        color: #2c3e50;
                    }}
                </style>
                <div style='
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    padding: 30px;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                '>
                    <div style='color: #1a73e8; font-size: 0.9em; font-weight: 500; margin-bottom: 20px;'>
                        VERIFICATION RESULTS
                    </div>
                    <div style='margin-bottom: 30px;'>
                        {html_content}
                    </div>
                    {sources_html}
                </div>
            """
            
            # Emit result
            self.analysis_complete.emit(final_html, citations)
            
        except Exception as e:
            self.analysis_error.emit(f"Error in analysis: {str(e)}")

    def analyze_with_llama(self, image_path):
        """Analyze screenshot using Together AI's Llama Vision model"""
        try:
            # Open and resize image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new dimensions while maintaining aspect ratio
                max_width = 800
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_width = max_width
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save resized image to bytes
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85, optimize=True)
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Prepare the prompt for content analysis
            prompt = "Analyze this screenshot and describe its content in detail. Focus on any claims, statements, or news being presented. What are the key points or claims being made?"
            
            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")

    def verify_with_perplexity(self, content):
        """Verify content using Perplexity API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You will be given a prompt with text giving you context about a news article, "
                        "news snippet or video. You have to understand the context from the prompt, "
                        "then do google search to verify the contents. Based on the web search results, "
                        "give response if the text is true or not. In both cases provide resources wherever "
                        "required. Always back your statement with relevant sources."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
            
            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages,
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"Error verifying content: {str(e)}")

    def get_video_id(self, url):
        """Extract video ID from YouTube URL (including Shorts)"""
        video_id = None
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$',
            r'(?:shorts\/)([0-9A-Za-z_-]{11}).*'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break
        return video_id

    def get_video_info(self, video_id):
        """Fetch video title and description from YouTube"""
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        html_content = response.text
        
        title_match = re.search(r'<meta name="title" content="(.*?)"', html_content)
        description_match = re.search(r'<meta name="description" content="(.*?)"', html_content)
        
        title = title_match.group(1) if title_match else "Title not found"
        description = description_match.group(1) if description_match else "Description not found"
        
        return {
            'title': html.unescape(title),
            'description': html.unescape(description)
        }

    def analyze_youtube(self, url):
        """Analyze YouTube video using transcript"""
        try:
            video_id = self.get_video_id(url)
            if not video_id:
                raise Exception("Invalid YouTube URL")

            # Get video transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            video_info = self.get_video_info(video_id)
            
            # Combine transcript text
            full_transcript = " ".join([entry['text'] for entry in transcript_list])
            
            # Prepare prompt for Perplexity
            prompt = f"""Video Title: {video_info['title']}
Description: {video_info['description']}

Transcript:
{full_transcript}

Please analyze this YouTube video content and:
1. Identify the main claims or statements made
2. Fact-check the key assertions
3. Look for any potential misinformation
4. Provide reliable sources to verify or dispute claims
5. Give an overall assessment of the video's credibility

Focus on verifying factual claims and identifying any misleading information.

Format your response with clear sections:
1. Main Claims
2. Fact-Check Results
3. Sources (provide numbered list of sources)
4. Overall Assessment"""

            # Send to Perplexity for analysis
            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fact-checker specializing in analyzing YouTube content for misinformation. Always provide numbered sources at the end of your analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.7
            )

            analysis = response.choices[0].message.content

            # Extract citations from the analysis
            citations = []
            for line in analysis.split('\n'):
                if line.strip().startswith('[') and ']' in line:
                    citations.append(line.strip())

            # Convert analysis to HTML
            html_content = markdown2.markdown(analysis)
            
            # Create a card-like layout
            source_card = f"""
            <div class="source-card">
                <h3>YouTube Video Analysis</h3>
                <p><strong>Title:</strong> {video_info['title']}</p>
                <p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>
                <div class="analysis-content">
                    {html_content}
                </div>
            </div>
            """
            
            self.analysis_complete.emit(source_card, citations)

        except Exception as e:
            self.analysis_error.emit(f"Error analyzing YouTube video: {str(e)}")

    def get_video_id(self, url):
        """Extract video ID from YouTube URL (including Shorts)"""
        video_id = None
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$',
            r'(?:shorts\/)([0-9A-Za-z_-]{11}).*'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                break
        return video_id

    def get_video_info(self, video_id):
        """Fetch video title and description from YouTube"""
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        html_content = response.text
        
        title_match = re.search(r'<meta name="title" content="(.*?)"', html_content)
        description_match = re.search(r'<meta name="description" content="(.*?)"', html_content)
        
        title = title_match.group(1) if title_match else "Title not found"
        description = description_match.group(1) if description_match else "Description not found"
        
        return {
            'title': html.unescape(title),
            'description': html.unescape(description)
        }

class RecordingAnalysisWorker(QObject):
    """Worker class for analyzing recording screenshots"""
    analysis_complete = pyqtSignal(str, list)  # content, citations
    analysis_error = pyqtSignal(str)
    analysis_progress = pyqtSignal(str)

    def __init__(self, together_client, perplexity_client):
        super().__init__()
        self.together_client = together_client
        self.perplexity_client = perplexity_client

    def analyze_recording(self, recording_dir):
        """Analyze a sequence of screenshots from recording"""
        try:
            # Get list of screenshots in order
            screenshots = sorted([
                os.path.join(recording_dir, f) 
                for f in os.listdir(recording_dir) 
                if f.endswith('.png')
            ])
            
            if not screenshots:
                raise Exception("No screenshots found in recording")

            # Analyze each screenshot individually
            analyses = []
            for i, screenshot in enumerate(screenshots, 1):
                self.analysis_progress.emit(f"Analyzing screenshot {i} of {len(screenshots)}...")
                analysis = self.analyze_single_screenshot(screenshot)
                analyses.append(f"Screenshot {i} Analysis:\n{analysis}")

            # Combine all analyses
            combined_analysis = "\n\n".join(analyses)
            
            # Send combined analysis to Perplexity for verification
            self.analysis_progress.emit("Verifying claims and gathering sources...")
            verification_result = self.verify_with_perplexity(combined_analysis)
            
            # Extract content and citations
            content = verification_result.choices[0].message.content
            citations = verification_result.citations if hasattr(verification_result, 'citations') else []
            
            # Convert markdown to HTML
            html_content = markdown2.markdown(content)
            
            # Create source cards
            sources_html = self.create_source_cards(citations) if citations else ""
            
            # Create final HTML
            final_html = self.create_final_html(html_content, sources_html, len(screenshots))
            
            # Emit result
            self.analysis_complete.emit(final_html, citations)
            
        except Exception as e:
            self.analysis_error.emit(f"Error analyzing recording: {str(e)}")

    def analyze_single_screenshot(self, screenshot_path):
        """Analyze a single screenshot using Together AI"""
        try:
            # Open and resize image
            with Image.open(screenshot_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new dimensions while maintaining aspect ratio
                max_width = 800
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_width = max_width
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save resized image to bytes
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85, optimize=True)
                image_data = base64.b64encode(buffered.getvalue()).decode()

            prompt = f"""<image>{image_data}</image>
            You are an expert at analyzing content and identifying potential misinformation.
            Look at this screenshot and:
            1. Identify the main topic or claim being discussed
            2. Extract any factual claims or statements made
            3. Note any suspicious or questionable content
            4. Point out any potential signs of manipulation or misleading information
            
            Be thorough but concise in your analysis."""

            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fact-checker and content analyst, skilled at identifying potential misinformation and verifying claims."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error analyzing screenshot: {str(e)}")

    def verify_with_perplexity(self, content):
        """Verify content using Perplexity API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking expert. Your task is to analyze multiple screenshot analyses "
                        "and provide a comprehensive verification report. For the sequence:\n"
                        "1. Identify the overall narrative or topic progression\n"
                        "2. Verify all claims and statements across screenshots\n"
                        "3. Note any contradictions or changes in information\n"
                        "4. Provide evidence and reliable sources\n"
                        "5. Give an overall assessment of the content's reliability\n\n"
                        "Format your response in markdown with clear sections."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Please analyze and verify these screenshot analyses:\n\n{content}",
                },
            ]
            
            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages,
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"Error verifying with Perplexity: {str(e)}")

    def create_final_html(self, html_content, sources_html, num_screenshots):
        """Create final HTML with styling"""
        return f"""
            <style>
                * {{
                    box-sizing: border-box;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                }}
                p {{
                    color: #34495e;
                    line-height: 1.6;
                    margin-bottom: 1.2em;
                }}
                ul, ol {{
                    color: #34495e;
                    line-height: 1.6;
                    padding-left: 1.5em;
                }}
                li {{
                    margin-bottom: 0.5em;
                }}
                code {{
                    background: #f8f9fa;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-size: 0.9em;
                    color: #e83e8c;
                }}
                blockquote {{
                    border-left: 4px solid #1a73e8;
                    margin: 1.5em 0;
                    padding: 0.5em 1em;
                    background: #f8f9fa;
                    color: #2c3e50;
                }}
            </style>
            <div style='
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                padding: 30px;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            '>
                <div style='color: #1a73e8; font-size: 0.9em; font-weight: 500; margin-bottom: 20px;'>
                    RECORDING ANALYSIS ({num_screenshots} screenshots)
                </div>
                <div style='margin-bottom: 30px;'>
                    {html_content}
                </div>
                {sources_html}
            </div>
        """

    def create_source_cards(self, citations):
        """Create HTML for source cards"""
        sources_html = """
            <div style='margin-top: 30px; padding: 20px 0;'>
                <h3 style='color: #2c3e50; font-size: 1.3em; margin-bottom: 15px;'>Sources</h3>
                <div style='overflow-x: auto; white-space: nowrap; padding: 10px 0; margin: 0 -10px;'>
        """
        
        for i, citation in enumerate(citations, 1):
            domain = citation.split('/')[2] if len(citation.split('/')) > 2 else citation
            sources_html += f"""
                <div style='
                    display: inline-block;
                    width: 300px;
                    margin: 0 10px;
                    padding: 15px;
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    vertical-align: top;
                    transition: transform 0.2s, box-shadow 0.2s;
                    cursor: pointer;
                    white-space: normal;
                ' onmouseover='this.style.transform="translateY(-2px)";this.style.boxShadow="0 4px 8px rgba(0,0,0,0.15)"'
                   onmouseout='this.style.transform="none";this.style.boxShadow="0 2px 4px rgba(0,0,0,0.1)"'
                   onclick='window.open("{citation}", "_blank")'>
                    <div style='color: #1a73e8; font-weight: 500; margin-bottom: 8px;'>Source {i}</div>
                    <div style='color: #202124; font-size: 0.9em; margin-bottom: 10px;'>{domain}</div>
                    <div style='
                        color: #5f6368;
                        font-size: 0.8em;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        display: -webkit-box;
                        -webkit-line-clamp: 2;
                        -webkit-box-orient: vertical;
                    '>{citation}</div>
                </div>
            """
        
        sources_html += "</div></div>"
        return sources_html

class KeyboardHandler(QObject):
    screenshot_signal = pyqtSignal()
    toggle_recording_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.start_listener()

    def start_listener(self):
        # Define the key combinations
        COMBINATIONS = [
            {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode(char='u')},
            {keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode(char='y')}
        ]
        
        # The currently active keys
        current = set()

        def on_press(key):
            if key in {keyboard.Key.cmd, keyboard.Key.shift, 
                      keyboard.KeyCode(char='u'), keyboard.KeyCode(char='y')}:
                current.add(key)
                
                if any(all(k in current for k in combo) for combo in COMBINATIONS):
                    if keyboard.KeyCode(char='u') in current:
                        self.screenshot_signal.emit()
                    elif keyboard.KeyCode(char='y') in current:
                        self.toggle_recording_signal.emit()

        def on_release(key):
            try:
                current.remove(key)
            except KeyError:
                pass

        # Start the listener in a separate thread
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

class VerifAIed(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.screenshot_dir = "screenshots"
        self.recording_dir = os.path.join(self.screenshot_dir, "latest_recording")
        self.current_screenshot = os.path.join(self.screenshot_dir, "current_screenshot.png")
        self.video_dir = "videos"
        self.latest_video = os.path.join(self.video_dir, "latest_video.mp4")
        self.previous_screenshot = None  # Store previous screenshot for comparison
        self.similarity_threshold = 0.95  # SSIM threshold (95% similarity)
        self.screenshots_taken = 0  # Counter for recorded screenshots
        self.screenshots_skipped = 0  # Counter for skipped screenshots
        
        # Initialize API clients
        self.together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        self.perplexity_client = OpenAI(
            api_key=os.getenv('PERPLEXITY_API_KEY'),
            base_url="https://api.perplexity.ai"
        )
        
        # Initialize workers
        self.analysis_worker = AnalysisWorker(self.together_client, self.perplexity_client)
        self.recording_analysis_worker = RecordingAnalysisWorker(self.together_client, self.perplexity_client)
        
        # Connect signals
        self.analysis_worker.analysis_complete.connect(self.handle_analysis_complete)
        self.analysis_worker.analysis_error.connect(self.handle_analysis_error)
        self.analysis_worker.analysis_progress.connect(self.handle_analysis_progress)
        
        self.recording_analysis_worker.analysis_complete.connect(self.handle_analysis_complete)
        self.recording_analysis_worker.analysis_error.connect(self.handle_analysis_error)
        self.recording_analysis_worker.analysis_progress.connect(self.handle_analysis_progress)
        
        self.init_ui()
        self.setup_directories()
        
        # Setup recording timer
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self.take_recording_screenshot)
        
        # Setup keyboard handler
        self.keyboard_handler = KeyboardHandler()
        self.keyboard_handler.screenshot_signal.connect(self.take_screenshot)
        self.keyboard_handler.toggle_recording_signal.connect(self.toggle_recording)

    def handle_analysis_complete(self, html_content, citations):
        """Handle analysis completion"""
        self.source_display.setHtml(html_content)

    def handle_analysis_progress(self, progress_message):
        """Handle analysis progress update"""
        progress_html = f"""
            <div style='
                font-family: Arial, sans-serif;
                padding: 20px;
                color: #0c5460;
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 8px;
                margin: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            '>
                <div style='display: flex; align-items: center;'>
                    <div style='
                        width: 20px;
                        height: 20px;
                        border: 2px solid #0c5460;
                        border-top-color: transparent;
                        border-radius: 50%;
                        margin-right: 10px;
                        animation: spin 1s linear infinite;
                    '></div>
                    <p style='margin: 0;'>{progress_message}</p>
                </div>
            </div>
            <style>
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
        """
        self.source_display.setHtml(progress_html)

    def handle_analysis_error(self, error_message):
        """Handle analysis error"""
        error_html = f"""
            <div style='
                font-family: Arial, sans-serif;
                padding: 20px;
                color: #721c24;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 4px;
            '>
                <div style='display: flex; align-items: center;'>
                    <div style='
                        color: #dc3545;
                        font-size: 24px;
                        margin-right: 10px;
                    '>⚠️</div>
                    <div>
                        <h3 style='margin: 0 0 10px 0; color: #721c24;'>Error</h3>
                        <p style='margin: 0;'>{error_message}</p>
                    </div>
                </div>
            </div>
        """
        self.source_display.setHtml(error_html)

    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle('VerifAIed')
        self.setGeometry(100, 100, 800, 600)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Create buttons
        button_layout = QHBoxLayout()
        
        self.screenshot_btn = QPushButton('Take Screenshot (⌘+⇧+S)')
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        button_layout.addWidget(self.screenshot_btn)
        
        self.record_btn = QPushButton('Start Recording (⌘+⇧+R)')
        self.record_btn.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_btn)
        
        self.youtube_btn = QPushButton('Analyze YouTube')
        self.youtube_btn.clicked.connect(self.analyze_youtube_video)
        button_layout.addWidget(self.youtube_btn)
        
        layout.addLayout(button_layout)

        # Create display area
        self.source_display = QTextEdit()
        self.source_display.setReadOnly(True)
        layout.addWidget(self.source_display)

    def setup_directories(self):
        """Setup all required directories"""
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.recording_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def play_feedback_sound(self):
        print('\a')  # Print ASCII bell character for system beep
        sys.stdout.flush()

    def take_screenshot(self):
        """Take a screenshot and analyze it"""
        try:
            # Play feedback sound first
            self.play_feedback_sound()
            
            with mss.mss() as sct:
                # Get the first monitor
                monitor = sct.monitors[1]
                
                # Take the screenshot
                screenshot = sct.grab(monitor)
                
                # Ensure screenshot directory exists
                os.makedirs(self.screenshot_dir, exist_ok=True)
                
                # Save screenshot
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=self.current_screenshot)
                
                self.source_display.setHtml("""
                    <div style='font-family: Arial, sans-serif; padding: 20px; color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;'>
                        <p>Screenshot captured! Analyzing content...</p>
                    </div>
                """)
                
                # Start analysis in worker
                self.analysis_worker.analyze_screenshot(self.current_screenshot)
                
        except Exception as e:
            self.handle_analysis_error(f"Error taking screenshot: {str(e)}")

    def compare_screenshots(self, current_img):
        """Compare current screenshot with previous one using SSIM"""
        if self.previous_screenshot is None:
            return False  # No previous screenshot, save this one
        
        try:
            # Convert to grayscale and calculate SSIM
            similarity = ssim(self.previous_screenshot, current_img, 
                            data_range=current_img.max() - current_img.min())
            
            # Return True if images are too similar (should skip)
            return similarity > self.similarity_threshold
        except Exception as e:
            print(f"Error comparing screenshots: {e}")
            return False  # On error, save the screenshot to be safe

    def take_recording_screenshot(self):
        """Take screenshots during recording session with similarity check"""
        with mss.mss() as sct:
            # Capture screenshot
            screenshot = sct.grab(sct.monitors[1])  # Primary monitor
            
            # Convert to numpy array and grayscale
            current_img = np.array(screenshot)
            current_gray = np.mean(current_img, axis=2)  # Convert to grayscale
            
            # Check if this screenshot is different enough
            if not self.compare_screenshots(current_gray):
                # Save screenshot
                output_path = os.path.join(
                    self.recording_dir,
                    f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                )
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
                
                # Update previous screenshot and counter
                self.previous_screenshot = current_gray
                self.screenshots_taken += 1
                self.source_display.setHtml(f"""
                    <div style='font-family: Arial, sans-serif; padding: 20px; color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;'>
                        <p>Recording in progress... Saved: {self.screenshots_taken}, Skipped: {self.screenshots_skipped}</p>
                    </div>
                """)
            else:
                self.screenshots_skipped += 1
                self.source_display.setHtml(f"""
                    <div style='font-family: Arial, sans-serif; padding: 20px; color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;'>
                        <p>Recording in progress... Saved: {self.screenshots_taken}, Skipped: {self.screenshots_skipped} (No significant change)</p>
                    </div>
                """)

    def toggle_recording(self):
        self.play_feedback_sound()
        if not self.recording:
            # Clear previous recording and start new session
            self.clear_recording_directory()
            self.previous_screenshot = None
            self.screenshots_taken = 0
            self.screenshots_skipped = 0
            self.recording = True
            self.record_btn.setText('Stop Recording (⌘+⇧+R)')
            self.recording_timer.start(1000)  # Start timer with 1-second interval
            self.source_display.setHtml("""
                <div style='font-family: Arial, sans-serif; padding: 20px; color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;'>
                    <p>Recording started... Screenshots will be saved when changes detected</p>
                </div>
            """)
        else:
            self.recording = False
            self.record_btn.setText('Start Recording (⌘+⇧+R)')
            self.recording_timer.stop()
            self.previous_screenshot = None
            
            # Start analysis if screenshots were taken
            if self.screenshots_taken > 0:
                self.recording_analysis_worker.analyze_recording(self.recording_dir)
            else:
                self.source_display.setHtml("""
                    <div style='font-family: Arial, sans-serif; padding: 20px; color: #856404; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 4px;'>
                        <p>Recording stopped. No screenshots were captured.</p>
                    </div>
                """)

    def clear_recording_directory(self):
        """Clear the contents of the recording directory"""
        if os.path.exists(self.recording_dir):
            shutil.rmtree(self.recording_dir)
        os.makedirs(self.recording_dir)

    def analyze_youtube_video(self):
        """Analyze YouTube video using transcript"""
        url, ok = QInputDialog.getText(self, 'Enter YouTube URL', 
                                     'Please enter the YouTube URL:')
        
        if ok and url:
            try:
                self.source_display.setHtml("""
                    <div style='font-family: Arial, sans-serif; padding: 20px; color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;'>
                        <p>Initializing analysis... Please wait</p>
                    </div>
                """)
                QApplication.processEvents()
                
                # Create a thread for analysis
                analysis_thread = Thread(target=self.analysis_worker.analyze_youtube, args=(url,))
                analysis_thread.daemon = True
                analysis_thread.start()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error starting analysis: {str(e)}")
                self.source_display.setHtml(f"""
                    <div style='font-family: Arial, sans-serif; padding: 20px; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;'>
                        <h3>Error</h3>
                        <p>{str(e)}</p>
                    </div>
                """)

    def closeEvent(self, event):
        # Stop the keyboard listener
        if hasattr(self.keyboard_handler, 'listener'):
            self.keyboard_handler.listener.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = VerifAIed()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
