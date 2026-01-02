from dotenv import load_dotenv
load_dotenv()
import argparse
import os
from src.webui.interface import theme_map, create_ui


def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--password", type=str, default=None, help="Password for web UI access (overrides WEBUI_PASSWORD env var)")
    args = parser.parse_args()

    # Get password from CLI argument or environment variable
    password = args.password or os.getenv("WEBUI_PASSWORD", "")
    
    # Configure authentication if password is set
    auth = None
    if password:
        # Password-only authentication using a callable function
        # The function ignores username and only validates password
        def check_password(username, pwd):
            # Ignore username, only check password
            return pwd == password
        
        auth = check_password
    
    demo = create_ui(theme_name=args.theme)
    demo.queue().launch(server_name=args.ip, server_port=args.port, auth=auth)


if __name__ == '__main__':
    main()
