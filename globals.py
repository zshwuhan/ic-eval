#s3_input_path = "s3://joeloren//iceval_out//input//datasets//"
tmp_dir_out = "s3://joeloren/interim_out/"
tmp_dir_in = "s3://joeloren/interim_in/"
tmp_dir_in_relative = "interim_in/"
tmp_dir_out_relative = "interim_out/"

from mrjob.protocol import JSONValueProtocol, JSONProtocol
jvp = JSONValueProtocol()
jp = JSONProtocol()

from boto.s3.connection import S3Connection
import sys

c = S3Connection('AKIAI4OZ3HY56BTOHA3A',
                 '6isbkZjBM8kt3PIk53EXVIf76VOPxOH8rNleGc6B')

bucket = c.get_bucket("joeloren")
datasets_bucket = c.get_bucket('joel_datasets')
