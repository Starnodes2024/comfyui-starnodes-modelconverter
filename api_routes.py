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
        # match_info is URL-decoded, so %2F / %5C arrive as path separators and
        # '..' walked straight out of PROFILES_DIR into any file on disk.
        if (not profile_name or profile_name in (".", "..")
                or os.path.basename(profile_name) != profile_name):
            return web.json_response({"error": "Invalid profile name"}, status=400)
        profile_path = os.path.join(PROFILES_DIR, profile_name)
        
        if not os.path.exists(profile_path):
            return web.json_response({"error": "Profile not found"}, status=404)
        
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
        
        # Return only metadata (not the full layer list)
        metadata = profile_data.get("__metadata__", {})
        
        return web.json_response({"__metadata__": metadata})
    
    except Exception as e:
        # the raw detail echoes filesystem paths and parser positions back to
        # the caller; keep it in the console, return a generic error
        print(f"[StarNodes] profile read failed: {e}")
        return web.json_response({"error": "Could not read profile"}, status=500)


print("[StarNodes] Profile API routes registered")
