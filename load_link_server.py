#!/usr/bin/python
from link_server import LinkServerCP
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-cp', type = str)

if __name__ == "__main__":
    parameters = parser.parse_args()
    print "name of dataset file: ", parameters.cp
    L = LinkServerCP(parameters.cp)
    
