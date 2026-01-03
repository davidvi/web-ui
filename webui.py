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
    parser.add_argument("--username", type=str, default=None, help="Username for web UI access (overrides WEBUI_USERNAME env var)")
    parser.add_argument("--password", type=str, default=None, help="Password for web UI access (overrides WEBUI_PASSWORD env var)")
    args = parser.parse_args()

    # Get username and password from CLI arguments or environment variables
    username = args.username or os.getenv("WEBUI_USERNAME", "")
    password = args.password or os.getenv("WEBUI_PASSWORD", "")
    
    # Configure authentication if both username and password are set
    auth = None
    if username and password:
        # Username and password authentication
        def check_auth(user, pwd):
            return user == username and pwd == password
        
        auth = check_auth
    elif password:
        # Fallback to password-only if only password is set
        def check_password(user, pwd):
            # Ignore username, only check password
            return pwd == password
        
        auth = check_password
    
    demo = create_ui(theme_name=args.theme)
    demo.queue().launch(server_name=args.ip, server_port=args.port, auth=auth)


if __name__ == '__main__':
    main()
