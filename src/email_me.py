from __future__ import unicode_literals
from __future__ import print_function
import sys
import argparse
import requests
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="EC2 results", help="Subject line of the email")
    return parser.parse_args()


def main():
    args = parse_args()
    send_simple_message("".join(sys.stdin.readlines()), args.subject)


def send_simple_message(text, subject="EC2 results"):
    return requests.post(
        "https://api.mailgun.net/v3/sandbox6797064448c345bfbbb9684adc854e7f.mailgun.org/messages",
        auth=("api", os.environ["MAILGUN_KEY"]),
        data={"from": "Mailgun Sandbox <postmaster@sandbox6797064448c345bfbbb9684adc854e7f.mailgun.org>",
              "to": ["Keith Trnka <keith.trnka@gmail.com>"],
              "subject": subject,
              "text": text})


if __name__ == "__main__":
    sys.exit(main())