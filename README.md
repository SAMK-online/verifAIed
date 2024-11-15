# VerifAIed

A powerful desktop application for verifying online content and detecting misinformation using advanced AI technologies.

## Features

### 1. Screenshot Analysis
- Instantly analyze any content on your screen
- Get real-time fact-checking and verification
- Receive detailed analysis with source citations
- Keyboard shortcut: ⌘+⇧+S

### 2. Recording Analysis
- Record a sequence of screen changes
- Automatically captures when content changes
- Analyzes the entire sequence for comprehensive verification
- Perfect for analyzing dynamic content or scrolling feeds
- Keyboard shortcut: ⌘+⇧+R

### 3. YouTube Analysis
- Analyze YouTube videos for misinformation
- Transcribes and fact-checks video content
- Provides comprehensive analysis of claims made
- Verifies statements against reliable sources
- Works with regular YouTube videos and Shorts

## Technology Stack

- **Frontend**: PyQt5 for native desktop interface
- **Image Processing**: PIL for image optimization
- **AI Analysis**: 
  - Together AI (Llama Vision) for image analysis
  - Perplexity API for fact verification
- **Video Processing**: youtube-transcript-api for transcription
- **Output**: Markdown to HTML rendering

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
TOGETHER_API_KEY=your_together_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## Usage

1. Launch the application:
```bash
python main.py
```

2. Choose your analysis method:
   - **Screenshot**: Click "Take Screenshot" or press ⌘+⇧+S
   - **Recording**: Click "Start Recording" or press ⌘+⇧+R
   - **YouTube**: Click "Analyze YouTube" and enter video URL

3. View the analysis results with:
   - Main claims identified
   - Fact-checking results
   - Source citations
   - Overall credibility assessment

## Requirements

- Python 3.8+
- macOS (primary development platform)
- API keys for Together AI and Perplexity
- Internet connection for AI analysis

## Notes

- Screenshot and recording analysis use Together AI's Llama Vision model
- Fact verification performed by Perplexity's Llama-3.1-Sonar model
- YouTube analysis uses video transcripts for comprehensive verification
- All analyses include source citations for verification

## Privacy & Security

- API keys stored securely in .env file
- No data stored permanently
- All analysis performed in real-time
- No user data collection

## Future Development

- Enhanced error handling
- Additional video platform support
- Improved source verification
- Advanced AI model integration

For issues or suggestions, please open a GitHub issue.
