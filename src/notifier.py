"""
ImpellerVision — Ziyaret Bildirimi (Gmail SMTP)
===============================================
Demoya biri girince e-posta yollar. Kimlik bilgileri ortam degiskenlerinden
(.env, gitignore'lu) okunur; KOD ICINDE GIZLI BILGI YOKTUR.

Beklenen ortam degiskenleri:
    SMTP_HOST       (orn. smtp.gmail.com)
    SMTP_PORT       (orn. 587)
    SMTP_USER       (gmail adresin)
    SMTP_PASSWORD   (Gmail UYGULAMA SIFRESI — normal sifre degil)
    NOTIFY_TO       (bildirimin gidecegi adres; bos ise SMTP_USER)
    SMTP_FROM_NAME  (gonderen adi; varsayilan ImpellerVision)
    NOTIFY_DEDUPE_SECONDS (ayni IP'ye tekrar yollamadan once bekleme; varsayilan 21600 = 6 saat)

Spam korumasi: IP basina dedupe + (frontend) oturumda tek cagri + arka planda gonderim.
"""
import os
import smtplib
import ssl
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

# Turkiye sabit UTC+3 (yil boyu, DST yok) — tzdata bagimliligi olmadan
TR_TZ = timezone(timedelta(hours=3))

# Kalici dedupe: her ziyaretci (tarayici ID) icin OMUR BOYU tek e-posta.
# Gorulen ID'ler bir dosyada saklanir (Docker'da volume -> restart'ta kaybolmaz).
_STORE = Path(os.getenv("NOTIFY_STORE")
               or (Path(__file__).resolve().parent.parent / "notify_data" / "seen_visitors.txt"))
_lock = threading.Lock()
_seen = None  # lazy yuklenen set

_last_sent = {}
_lock = threading.Lock()


def _cfg():
    return {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER"),
        # Gmail uygulama sifresi bosluklu gosterilir; bosluklari at
        "password": (os.getenv("SMTP_PASSWORD") or "").replace(" ", ""),
        "to": os.getenv("NOTIFY_TO") or os.getenv("SMTP_USER"),
        "from_name": os.getenv("SMTP_FROM_NAME", "ImpellerVision"),
    }


def is_configured() -> bool:
    c = _cfg()
    return bool(c["host"] and c["user"] and c["password"])


def _load_seen():
    global _seen
    if _seen is None:
        _seen = set()
        try:
            if _STORE.exists():
                _seen = {ln.strip() for ln in _STORE.read_text(encoding="utf-8").splitlines() if ln.strip()}
        except Exception as e:
            print(f"[notify] dedupe dosyasi okunamadi: {e}")
    return _seen


def should_send(key: str) -> bool:
    """Her ziyaretci (tarayici ID; yoksa IP) icin OMUR BOYU tek sefer True doner.
    Gorulen anahtarlar kalici olarak dosyaya yazilir."""
    with _lock:
        seen = _load_seen()
        if key in seen:
            return False
        seen.add(key)
        try:
            _STORE.parent.mkdir(parents=True, exist_ok=True)
            with open(_STORE, "a", encoding="utf-8") as f:
                f.write(key + "\n")
        except Exception as e:
            print(f"[notify] dedupe dosyasina yazilamadi: {e}")
        return True


def send_visit_notification(ip: str, user_agent: str = "?", referer: str = "-", path: str = "/"):
    """Senkron e-posta gonderimi (FastAPI BackgroundTasks ile arka planda cagrilir)."""
    if not is_configured():
        return
    c = _cfg()
    msg = EmailMessage()
    msg["Subject"] = f"🔔 ImpellerVision — yeni ziyaretçi ({ip})"
    msg["From"] = f"{c['from_name']} <{c['user']}>"
    msg["To"] = c["to"]
    msg.set_content(
        "ImpellerVision demosuna yeni bir ziyaret geldi.\n\n"
        f"IP        : {ip}\n"
        f"Zaman     : {datetime.now(TR_TZ).strftime('%Y-%m-%d %H:%M:%S')} (TR)\n"
        f"Sayfa     : {path}\n"
        f"Referrer  : {referer}\n"
        f"Tarayıcı  : {user_agent}\n"
    )
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(c["host"], c["port"], timeout=10) as s:
            s.starttls(context=ctx)
            s.login(c["user"], c["password"])
            s.send_message(msg)
        print(f"[notify] ziyaret bildirimi gonderildi -> {c['to']} (ip={ip})")
    except Exception as e:
        print(f"[notify] e-posta gonderilemedi: {e}")
