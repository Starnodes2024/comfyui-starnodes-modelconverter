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

        # profile_name comes straight from the URL. Reject anything that isn't
        # a plain "*.json" filename so a crafted (possibly percent-encoded)
        # value like "../../secrets.json" can't escape PROFILES_DIR.
        safe_name = os.path.basename(profile_name)
        if safe_name != profile_name or not safe_name.endswith(".json"):
            return web.json_response({"error": "Invalid profile name"}, status=400)

        profiles_root = os.path.realpath(PROFILES_DIR)
        profile_path = os.path.realpath(os.path.join(profiles_root, safe_name))
        if os.path.commonpath([profile_path, profiles_root]) != profiles_root:
            return web.json_response({"error": "Invalid profile name"}, status=400)

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
