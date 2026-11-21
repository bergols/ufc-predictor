# Bloqueia toda rede no processo de teste.
import socket
def _blocked(*a, **k):
    raise OSError("CHAOS: rede bloqueada no ambiente isolado")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
