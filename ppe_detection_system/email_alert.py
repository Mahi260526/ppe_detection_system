"""
Email alerts for PPE violations (non-blocking).
Based on: https://github.com/Ansarimajid/Construction-PPE-Detection
Configure SENDER_EMAIL, RECEIVER_EMAIL, EMAIL_PASSWORD in .env to enable.
"""
import os
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

def is_email_configured():
    """Return True if email credentials are set in .env"""
    _load_env()
    return bool(os.getenv("SENDER_EMAIL") and os.getenv("RECEIVER_EMAIL") and os.getenv("EMAIL_PASSWORD"))

def send_email_alert(image_path, video_source="CCTV", violation_tags=None, location=None, datetime_str=None,
                     recipient_name=None, company_name=None):
    """
    Send an email with the violation image attached.
    violation_tags: list of strings e.g. ["No Helmet", "No Vest"]
    location: optional location name for the camera/site.
    datetime_str: violation date & time e.g. "2025-02-12 14:30:45"
    recipient_name: for salutation (default from env RECIPIENT_NAME or "Recipient")
    company_name: for sign-off (default from env COMPANY_NAME or "Company")
    """
    _load_env()
    sender = os.getenv("SENDER_EMAIL")
    receiver = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("EMAIL_PASSWORD")
    if not sender or not receiver or not password:
        return
    violation_tags = violation_tags or ["PPE violation"]
    category_str = violation_tags[0] if violation_tags else "PPE violation"
    tags_str = ", ".join(violation_tags)
    location_str = (location or "").strip() or "Not specified"
    datetime_display = (datetime_str or "").strip() or "N/A"
    recipient = (recipient_name or os.getenv("RECIPIENT_NAME") or "Recipient").strip()
    company = (company_name or os.getenv("COMPANY_NAME") or "Company").strip()

    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = f"🚨 PPE Violation Alert – {category_str} Detected | {location_str}"

    body = (
        f"Dear {recipient},\n\n"
        "A PPE violation has been detected by the monitoring system. Please find the details below:\n\n"
        "Violation Details:\n\n"
        f"Category: {tags_str}\n"
        f"Location: {location_str}\n"
        f"Camera ID: {video_source}\n"
        f"Date & Time: {datetime_display}\n\n"
        "An image capturing the violation is attached for your reference.\n\n"
        "Kindly review and take necessary corrective action.\n\n"
        "If this alert appears to be incorrect, please inform the system administrator.\n\n"
        "Regards,\n"
        "Safety Monitoring System\n"
        f"{company}"
    )
    message.attach(MIMEText(body, "plain"))
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
        message.attach(part)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, message.as_string())
        print("Email alert sent with attachment.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_email_in_background(image_path, video_source="CCTV", violation_tags=None, location=None,
                             datetime_str=None, recipient_name=None, company_name=None):
    """Send email in a background thread so the video feed is not blocked."""
    thread = threading.Thread(
        target=send_email_alert,
        args=(image_path,),
        kwargs={
            "video_source": video_source,
            "violation_tags": violation_tags,
            "location": location,
            "datetime_str": datetime_str,
            "recipient_name": recipient_name,
            "company_name": company_name,
        },
    )
    thread.daemon = True
    thread.start()
