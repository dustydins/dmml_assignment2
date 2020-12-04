#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for pretty printing text to terminal
"""

from termcolor import colored


def cprint(text, colour="white"):
    """
    Conveniance for printing pretty text
    """
    print(colored(text, colour))


def print_header(text, colour="white"):
    """
    Conveniance for printing pretty header
    """
    print(colored("====================================================================================", colour))
    cprint(text, colour)
    print(colored("====================================================================================", colour))


def print_footer(colour="white"):
    """
    Conveniance for printing pretty footer
    """
    print(colored("====================================================================================", colour))
    print(colored("////////////////////////////////////////////////////////////////////////////////////\n", colour))


def print_div(colour="white"):
    """
    Conveniance for printing pretty divider
    """
    print(colored("------------------------------------------------------------------------------------", colour))
