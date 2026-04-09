TOOLS = [
    {
        "name": "web_search",
        "description": "Search the internet for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_image",
        "description": "Generate an image from a prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"}
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "edit_image",
        "description": "Modify the last generated image.",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string"}
            },
            "required": ["instruction"],
        },
    },
]
