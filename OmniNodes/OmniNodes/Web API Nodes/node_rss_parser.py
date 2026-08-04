# ComfyUI_WebAPI_Nodes/node_rss_parser.py
import requests
import re
from typing import List, Dict

class RSSFeedParserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feed_url": ("STRING", {"default": "https://feeds.feedburner.com/PythonInsider"}),
                "max_items": ("INT", {"default": 5, "min": 1, "max": 50}),
            },
            "optional": {
                "filter_keyword": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("entries", "raw_xml")
    FUNCTION = "parse_feed"
    CATEGORY = "TensorVizion/Web API"
    DESCRIPTION = "Fetch and parse RSS/Atom feed; returns list of entries."

    def parse_feed(self, feed_url: str, max_items: int, filter_keyword: str = ""):
        try:
            resp = requests.get(feed_url, timeout=10)
            resp.raise_for_status()
            xml_content = resp.text

            items = re.findall(r"<item>(.*?)</item>", xml_content, re.DOTALL)
            entries = []

            for i, item in enumerate(items):
                if i >= max_items:
                    break

                title = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
                link = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
                desc = re.search(r"<description>(.*?)</description>", item, re.DOTALL)
                pub_date = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)

                entry = {
                    "title": title.group(1).strip() if title else "",
                    "link": link.group(1).strip() if link else "",
                    "summary": desc.group(1).strip() if desc else "",
                    "pub_date": pub_date.group(1).strip() if pub_date else "",
                }

                if filter_keyword and filter_keyword.lower() not in (
                    entry["title"].lower() + entry["summary"].lower()
                ):
                    continue

                entries.append(entry)

            return (entries, xml_content)
        except Exception as e:
            return ([], f"Error: {str(e)}")

# --- Define mappings for OmniNodes ---
NODE_CLASS_MAPPINGS = {
    "RSSFeedParserNode": RSSFeedParserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RSSFeedParserNode": "RSS Feed Parser (WebAPI)"
}
# ---