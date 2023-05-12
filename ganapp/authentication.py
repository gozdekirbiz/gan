from django.core.mail import send_mail
import random

def send_verification_email(email, verification_code):
    subject = 'Verification Code'
    message = f"Verification Code: {verification_code}"
    from_email = 'musicgeneratorgazi@gmail.com'
    recipient_list = [email]

    send_mail(subject, message, from_email, recipient_list)


def generate_verification_code():
    # Rastgele 6 haneli bir doğrulama kodu oluştur
    verification_code = random.randint(100000, 999999)
    return str(verification_code)
