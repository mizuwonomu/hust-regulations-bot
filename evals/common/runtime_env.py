"""Nạp biến môi trường runtime theo yêu cầu, không load gì lúc import module.

Import `dotenv` nằm trong hàm nên module này an toàn để import ở mọi script, kể
cả các đường offline. Chỉ khi caller truyền `env_file` thì file mới được đọc.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_ENV_FILE = ".env"


def load_runtime_env(env_file: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Nạp một env file nếu được chỉ định và trả path đã nạp.

    Params:
    - env_file: path tới env file; None nghĩa là không nạp gì, không tự đoán `.env`
    - override: mặc định False nên biến của process luôn thắng biến trong file

    Raise FileNotFoundError nếu path được chỉ định nhưng không tồn tại
    """
    if env_file is None:
        return None

    path = Path(env_file)
    if not path.exists():
        raise FileNotFoundError(f"{path}: env file not found")

    #Import provider chỉ khi thực sự được gọi để import module không kéo dotenv
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=str(path), override=override)
    return path


__all__ = ["DEFAULT_ENV_FILE", "load_runtime_env"]
