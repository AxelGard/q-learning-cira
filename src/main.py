#!/usr/bin/env python3
import cira
import random
import time

"""
Description of your project
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"


cira.alpaca.KEY_FILE = "../test_key.json"

portfolio = cira.Portfolio()
exchange = cira.Exchange()


def main():  
    print(exchange.is_open)


if __name__ == "__main__":
    main()