"""API routes for Star Model Converter Pro - Profile metadata endpoint."""
import os
import json
from aiohttp import web
import server

# Profile directory
PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")


@server.PromptServer.instance.routes.get("/starnodes/profile/{profile_name}")
async def get_profile_metadata(request):
    """Return profile metadata for tooltip display."""
    try:
        profile_name = request.match_info["profile_name"]
        profile_path = os.path.join(PROFILES_DIR, profile_name)
        
        if not os.path.exists(profile_path):
            return web.json_response({"error": "Profile not found"}, status=404)
        
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        
        # Return only metadata (not the full layer list)
        metadata = profile_data.get("__metadata__", {})
        
        return web.json_response({"__metadata__": metadata})
    
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


print("[StarNodes] Profile API routes registered")
