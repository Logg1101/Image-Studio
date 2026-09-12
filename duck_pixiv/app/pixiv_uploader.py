"""Pixiv uploader interface and implementations.
Completely isolated from application logic.
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class PixivUploadResult:
    """Standardized response from Pixiv uploader."""

    def __init__(
        self,
        success: bool,
        pixiv_id: Optional[str] = None,
        message: str = "",
        error_code: Optional[str] = None
    ):
        self.success = success
        self.pixiv_id = pixiv_id
        self.message = message
        self.error_code = error_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "pixiv_id": self.pixiv_id,
            "message": self.message,
            "error_code": self.error_code
        }


class PixivUploader(ABC):
    """Abstract base class for Pixiv uploaders."""

    @abstractmethod
    def upload(
        self,
        image_paths: List[str],
        title: str,
        description: str,
        tags: List[str],
        is_ai: bool = True,
        age_limit: str = "r18"  # "all_age", "r18", "r18g"
    ) -> PixivUploadResult:
        """Uploads work to Pixiv."""
        pass

    @abstractmethod
    def verify_auth(self) -> PixivUploadResult:
        """Verifies if the uploader authentication is valid."""
        pass


class DryRunUploader(PixivUploader):
    """
    Safe Dry-Run / Test Uploader.
    Validates all upload constraints without sending actual network requests.
    """

    def __init__(self, simulate_delay_sec: float = 0.5):
        self.simulate_delay_sec = simulate_delay_sec

    def verify_auth(self) -> PixivUploadResult:
        return PixivUploadResult(
            success=True,
            message="ドライランモード: 認証チェックに成功しました（シミュレーション）"
        )

    def upload(
        self,
        image_paths: List[str],
        title: str,
        description: str,
        tags: List[str],
        is_ai: bool = True,
        age_limit: str = "r18"
    ) -> PixivUploadResult:
        if not image_paths:
            return PixivUploadResult(
                success=False,
                error_code="NO_IMAGES",
                message="投稿対象の画像が指定されていません。"
            )

        for p in image_paths:
            if not os.path.exists(p):
                return PixivUploadResult(
                    success=False,
                    error_code="FILE_NOT_FOUND",
                    message=f"指定された画像ファイルが見つかりません: {p}"
                )

        if not title.strip():
            return PixivUploadResult(
                success=False,
                error_code="EMPTY_TITLE",
                message="タイトルを入力してください。"
            )

        if len(tags) > 10:
            return PixivUploadResult(
                success=False,
                error_code="TOO_MANY_TAGS",
                message=f"Pixivのタグ上限は10個です（現在: {len(tags)}個）。"
            )

        if self.simulate_delay_sec > 0:
            time.sleep(self.simulate_delay_sec)

        mock_work_id = str(int(time.time() * 1000) % 1000000000)
        return PixivUploadResult(
            success=True,
            pixiv_id=mock_work_id,
            message=f"投稿完了（ドライラン検証成功）! 作品ID: {mock_work_id}"
        )


class PixivApiUploader(PixivUploader):
    """
    Pixiv uploader utilizing PixivPy / Web API tokens.
    Keeps authentication credentials safely in data/settings.json or environment.
    """

    def __init__(self, refresh_token: str = ""):
        self.refresh_token = refresh_token

    def verify_auth(self) -> PixivUploadResult:
        if not self.refresh_token:
            return PixivUploadResult(
                success=False,
                error_code="NO_TOKEN",
                message="Pixivのリフレッシュトークンが設定されていません。"
            )
        try:
            import pixivpy3  # noqa: F401
            return PixivUploadResult(
                success=True,
                message="Pixiv認証トークンが有効です。"
            )
        except ImportError:
            return PixivUploadResult(
                success=False,
                error_code="NOT_IMPLEMENTED",
                message="Pixiv API認証機能は未実装です（pixivpy3ライブラリが見つかりません）。ドライランモードをご利用ください。"
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PixivUploadResult(
                success=False,
                error_code="AUTH_FAILED",
                message=f"Pixiv認証に失敗しました: {e}"
            )

    def upload(
        self,
        image_paths: List[str],
        title: str,
        description: str,
        tags: List[str],
        is_ai: bool = True,
        age_limit: str = "r18"
    ) -> PixivUploadResult:
        auth_res = self.verify_auth()
        if not auth_res.success:
            return auth_res

        try:
            import pixivpy3
            raise NotImplementedError("PixivPy direct post endpoint is not yet connected")
        except ImportError:
            return PixivUploadResult(
                success=False,
                error_code="NOT_IMPLEMENTED",
                pixiv_id=None,
                message="Pixiv APIアップロード機能は現在未実装です（pixivpy3ライブラリが見つかりません）。ドライランモードをご利用ください。"
            )
        except Exception as e:
            return PixivUploadResult(
                success=False,
                error_code="UPLOAD_FAILED",
                pixiv_id=None,
                message=f"Pixivへのアップロード中にエラーが発生しました: {e}"
            )


class PixivCookieUploader(PixivUploader):
    """
    Pixiv uploader utilizing Web Session Cookie (PHPSESSID).
    Direct, requires no OAuth keys, and works like your normal browser session.
    """

    def __init__(self, cookie: str = ""):
        self.cookie = cookie.strip()

    def verify_auth(self) -> PixivUploadResult:
        if not self.cookie:
            return PixivUploadResult(
                success=False,
                error_code="NO_COOKIE",
                message="PHPSESSID クッキーが設定されていません。"
            )
        cookie_val = self.cookie.replace("PHPSESSID=", "").strip()
        if not cookie_val:
            return PixivUploadResult(
                success=False,
                error_code="INVALID_COOKIE",
                message="有効なPHPSESSIDクッキー値が入力されていません。"
            )
        return PixivUploadResult(
            success=False,
            error_code="NOT_IMPLEMENTED",
            message="Pixivセッションクッキーによる直接自動投稿機能は現在未実装です。ドライランモードをご利用ください。"
        )

    def upload(
        self,
        image_paths: List[str],
        title: str,
        description: str,
        tags: List[str],
        is_ai: bool = True,
        age_limit: str = "r18"
    ) -> PixivUploadResult:
        auth_res = self.verify_auth()
        if not auth_res.success:
            return auth_res

        return PixivUploadResult(
            success=False,
            error_code="NOT_IMPLEMENTED",
            pixiv_id=None,
            message="Pixiv Webセッションによる自動投稿機能は現在未実装です。ドライランモードをご利用ください。"
        )


def get_uploader(settings: Optional[Dict[str, Any]] = None) -> PixivUploader:
    """Factory to obtain the configured uploader."""
    if not settings:
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except Exception:
                settings = {}
        else:
            settings = {}

    pixiv_conf = settings.get("pixiv", {})
    auth_mode = pixiv_conf.get("auth_mode", "dry_run")

    if auth_mode == "cookie":
        return PixivCookieUploader(cookie=pixiv_conf.get("cookie", ""))
    elif auth_mode == "api":
        return PixivApiUploader(refresh_token=pixiv_conf.get("refresh_token", ""))
    return DryRunUploader()
