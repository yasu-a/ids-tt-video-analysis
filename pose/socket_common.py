import struct

# SOCKET_RECEIVE_ALL_BUFFER_SIZE = 1024 * 4

__all__ = 'send_blob', 'recv_blob'


# https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data

def send_blob(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_blob(sock):
    # Read message length and unpack it into an integer
    raw_msg_len = _recv_all(sock, 4)
    if not raw_msg_len:
        return None
    msg_len = struct.unpack('>I', raw_msg_len)[0]
    # Read the message data
    return _recv_all(sock, msg_len)


def _recv_all(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# def _socket_receive_all(s: socket.socket):
#     packet_lst = []
#     while True:
#         packet = s.recv(SOCKET_RECEIVE_ALL_BUFFER_SIZE)
#         packet_lst.append(packet)
#         if len(packet) == 0:
#             break
#     return b''.join(packet_lst)
