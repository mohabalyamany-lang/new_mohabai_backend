from app.services.web_search_service import web_search
from app.services.image_service import generate_image, edit_last_image


TOOL_REGISTRY = {
    "web_search": web_search,
    "generate_image": generate_image,
    "edit_image": edit_last_image,
}
