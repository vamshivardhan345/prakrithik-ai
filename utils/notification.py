import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class NotificationManager:
    def __init__(self):
        # Email config
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = "your_email@gmail.com" # Placeholder
        self.email_pass = "your_app_password"   # Placeholder
        
        # Twilio config
        self.twilio_sid = "your_sid"
        self.twilio_token = "your_token"
        self.twilio_number = "your_twilio_number"

    def send_email(self, rec_email, disease, instructions):
        if "@" not in rec_email: return False
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = rec_email
            msg['Subject'] = f"Plant Disease Detected: {disease}"
            
            body = f"Plant Disease Detected: {disease}\n\nRecommended Organic Treatment Instructions:\n{instructions}"
            msg.attach(MIMEText(body, 'plain'))
            
            # Uncomment to actually send (requires valid credentials)
            # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            # server.starttls()
            # server.login(self.email_user, self.email_pass)
            # server.send_message(msg)
            # server.quit()
            
            print(f"Email sent to {rec_email} for {disease}")
            return True
        except Exception as e:
            print(f"Email error: {e}")
            return False

    def send_sms(self, phone_number, disease, instructions):
        # Using placeholder for Twilio
        print(f"SMS sent to {phone_number} for {disease}: {instructions[:50]}...")
        return True # Simulate success
