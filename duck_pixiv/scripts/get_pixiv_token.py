"""Helper script to obtain Pixiv OAuth Refresh Token using standard PKCE flow."""
import sys
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, "data", "settings.json")


def save_token_to_settings(refresh_token: str, user_name: str = "", user_id: str = ""):
    """Conveniently saves the retrieved token into settings.json automatically."""
    try:
        settings = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)

        pixiv = settings.get("pixiv", {})
        pixiv["auth_mode"] = "api"
        pixiv["refresh_token"] = refresh_token
        if user_name or user_id:
            pixiv["user_id"] = user_name or user_id

        settings["pixiv"] = pixiv
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        print("\n[OK] Automatically saved token & account into duck_pixiv/data/settings.json!")
    except Exception as e:
        print(f"\n[Notice] Could not auto-save to settings.json: {e}")


def main():
    print("=" * 60)
    print(" Pixiv OAuth Refresh Token Helper")
    print("=" * 60)
    print("This will open the official Pixiv OAuth login page in your browser.")
    print("Log into Pixiv and complete the authorization prompt.")
    print("=" * 60)

    try:
        import gppt
    except ImportError:
        import subprocess
        print("[*] Installing helper 'gppt'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gppt"])
        import gppt

    try:
        # Uses gppt.oauth_login() which works in gppt 5.x
        print("\nOpening browser for Pixiv login...")
        token_obj = gppt.oauth_login(open_browser=True)

        if token_obj and hasattr(token_obj, "refresh_token"):
            token = token_obj.refresh_token
            user_name = getattr(token_obj, "user_name", "") or getattr(token_obj, "user_account", "")
            user_id = getattr(token_obj, "user_id", "")

            print("\n" + "=" * 60)
            print(" [SUCCESS] Pixiv Authentication Successful!")
            if user_name:
                print(f" Account: {user_name} (ID: {user_id})")
            print("\n Refresh Token:")
            print(f" {token}\n")
            print("=" * 60)

            # Auto-save so user doesn't even have to copy-paste manually!
            save_token_to_settings(token, user_name=user_name, user_id=user_id)
            print("\nYou're all set! You can close this window now.")

        else:
            print("[Error] Failed to retrieve token. Result:", token_obj)

    except Exception as e:
        print(f"\n[Error] {e}")
        print("\nAlternative: Check Developer Tools (F12 -> Network tab) when logging in at:")
        print("https://app-api.pixiv.net/web/v1/login")


if __name__ == "__main__":
    main()
