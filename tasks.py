# tasks.py
from celery import Celery
import smtplib
from email.mime.text import MIMEText
from flask import Flask

celery = Celery('tasks', backend='redis://localhost:6379', broker='redis://localhost:6379')

# sending emails asynchronously
@celery.task()
def send_email(to_email, subject, body):
    print("sending alert email..............")
      
    # SMTP server configuration for MailHog
    smtp_server = 'localhost'
    smtp_port = 1025  # MailHog's SMTP port

    # Create the email
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'voilencedetectiona@gmail.com'
    msg['To'] = to_email

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.sendmail(msg['From'], [msg['To']], msg.as_string())

    print("Email sent successfully.")

