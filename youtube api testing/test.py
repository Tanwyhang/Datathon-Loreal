from googleapiclient.discovery import build


# Replace with your API key
API_KEY = "AIzaSyBP0HwXe802_CC9tkO6Z19-nMrPQ_fU3AU"
VIDEO_ID = "tHiuBDhAOkQ"  # Example video ID

# Build YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# Fetch video metadata
request = youtube.videos().list(
    part="snippet,contentDetails,statistics",
    id=VIDEO_ID
)
response = request.execute()

# Extract metadata
if response["items"]:
    video = response["items"][0]
    snippet = video["snippet"]
    stats = video["statistics"]

    metadata = {
        "title": snippet["title"],
        "description": snippet["description"],
        "publishedAt": snippet["publishedAt"],
        "channelTitle": snippet["channelTitle"],
        "tags": snippet.get("tags", []),
        "duration": video["contentDetails"]["duration"],  # ISO 8601 format
        "viewCount": stats.get("viewCount"),
        "likeCount": stats.get("likeCount"),
        "commentCount": stats.get("commentCount")
    }

    print(metadata)
else:
    print("Video not found.")
