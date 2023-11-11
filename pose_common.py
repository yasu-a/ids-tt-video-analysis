import socket

SOCKET_RECEIVE_ALL_BUFFER_SIZE = 1024 * 4

__all__ = '_socket_receive_all',


def _socket_receive_all(s: socket.socket):
    packet_lst = []
    while True:
        packet = s.recv(SOCKET_RECEIVE_ALL_BUFFER_SIZE)
        packet_lst.append(packet)
        if len(packet) == 0:
            break
    return b''.join(packet_lst)
