"""Bounded socket framing for authenticated telemetry, usable with socketpair tests."""
import json
from .secure_channel import canonical
from .authenticated_handshake import AuthenticatedHandshake
MAX_FRAME=16*1024*1024


def read_exact(sock,size):
    data=bytearray()
    while len(data)<size:
        piece=sock.recv(size-len(data))
        if not piece:raise EOFError('Connection closed mid-frame')
        data.extend(piece)
    return bytes(data)


def send_frame(sock,data):
    if not 0<len(data)<=MAX_FRAME:raise ValueError('Frame size outside bounds')
    sock.sendall(len(data).to_bytes(4,'big')+data)


def receive_frame(sock):
    size=int.from_bytes(read_exact(sock,4),'big')
    if not 0<size<=MAX_FRAME:raise ValueError('Frame size outside bounds')
    return read_exact(sock,size)


def connect_authenticated(sock,handshake,initiator):
    if initiator:
        send_frame(sock,canonical(handshake.offer()))
        finish=handshake.finish(json.loads(receive_frame(sock)))
        send_frame(sock,canonical(finish))
    else:
        response=handshake.respond(json.loads(receive_frame(sock)))
        send_frame(sock,canonical(response))
        handshake.accept(json.loads(receive_frame(sock)))
    return handshake.channel(True),handshake.channel(False)


def process_telemetry(receiver,frame):
    value=json.loads(receiver.decrypt(frame))
    if not isinstance(value,dict) or value.get('type') not in {'alert','telemetry'}:
        raise ValueError('Unsupported telemetry payload')
    return value
