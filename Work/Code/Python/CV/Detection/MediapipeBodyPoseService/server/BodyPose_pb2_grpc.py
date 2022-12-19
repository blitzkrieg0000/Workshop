# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import BodyPose_pb2 as BodyPose__pb2


class BodyPoseStub(object):
    """Service
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ExtractBodyPose = channel.unary_unary(
                '/bodypose.BodyPose/ExtractBodyPose',
                request_serializer=BodyPose__pb2.ExtractBodyPoseRequest.SerializeToString,
                response_deserializer=BodyPose__pb2.ExtractBodyPoseResponse.FromString,
                )


class BodyPoseServicer(object):
    """Service
    """

    def ExtractBodyPose(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BodyPoseServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ExtractBodyPose': grpc.unary_unary_rpc_method_handler(
                    servicer.ExtractBodyPose,
                    request_deserializer=BodyPose__pb2.ExtractBodyPoseRequest.FromString,
                    response_serializer=BodyPose__pb2.ExtractBodyPoseResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'bodypose.BodyPose', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class BodyPose(object):
    """Service
    """

    @staticmethod
    def ExtractBodyPose(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/bodypose.BodyPose/ExtractBodyPose',
            BodyPose__pb2.ExtractBodyPoseRequest.SerializeToString,
            BodyPose__pb2.ExtractBodyPoseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)