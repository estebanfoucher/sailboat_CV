import re

def get_video_id_from_url(url):
    """
    Extract the YouTube video ID from a URL using regex.
    Returns the video ID as a string, or None if not found.
    """
    # Standard YouTube URL patterns
    patterns = [
        r'(?:v=|youtu\.be/|embed/|v/|shorts/)([\w-]{11})',
        r'youtube\.com/watch\?.*?v=([\w-]{11})',
        r'youtube\.com/embed/([\w-]{11})',
        r'youtu\.be/([\w-]{11})',
        r'youtube\.com/shorts/([\w-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None 