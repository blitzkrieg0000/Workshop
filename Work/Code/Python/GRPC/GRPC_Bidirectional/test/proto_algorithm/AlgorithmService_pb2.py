# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: AlgorithmService.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16\x41lgorithmService.proto\x12\rSamplePackage\"K\n\x17\x41lgorithmServiceRequest\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04\x63ode\x18\x02 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x03 \x01(\t\"l\n\x18\x41lgorithmServiceResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05title\x18\x02 \x01(\t\x12\x0c\n\x04\x63ode\x18\x03 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12\x12\n\ncreated_on\x18\x06 \x01(\x05\x32~\n\x10\x41lgorithmService\x12j\n\x11\x63reateBulkEntries\x12&.SamplePackage.AlgorithmServiceRequest\x1a\'.SamplePackage.AlgorithmServiceResponse\"\x00(\x01\x30\x01\x62\x06proto3')



_ALGORITHMSERVICEREQUEST = DESCRIPTOR.message_types_by_name['AlgorithmServiceRequest']
_ALGORITHMSERVICERESPONSE = DESCRIPTOR.message_types_by_name['AlgorithmServiceResponse']
AlgorithmServiceRequest = _reflection.GeneratedProtocolMessageType('AlgorithmServiceRequest', (_message.Message,), {
  'DESCRIPTOR' : _ALGORITHMSERVICEREQUEST,
  '__module__' : 'AlgorithmService_pb2'
  # @@protoc_insertion_point(class_scope:SamplePackage.AlgorithmServiceRequest)
  })
_sym_db.RegisterMessage(AlgorithmServiceRequest)

AlgorithmServiceResponse = _reflection.GeneratedProtocolMessageType('AlgorithmServiceResponse', (_message.Message,), {
  'DESCRIPTOR' : _ALGORITHMSERVICERESPONSE,
  '__module__' : 'AlgorithmService_pb2'
  # @@protoc_insertion_point(class_scope:SamplePackage.AlgorithmServiceResponse)
  })
_sym_db.RegisterMessage(AlgorithmServiceResponse)

_ALGORITHMSERVICE = DESCRIPTOR.services_by_name['AlgorithmService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _ALGORITHMSERVICEREQUEST._serialized_start=41
  _ALGORITHMSERVICEREQUEST._serialized_end=116
  _ALGORITHMSERVICERESPONSE._serialized_start=118
  _ALGORITHMSERVICERESPONSE._serialized_end=226
  _ALGORITHMSERVICE._serialized_start=228
  _ALGORITHMSERVICE._serialized_end=354
# @@protoc_insertion_point(module_scope)
