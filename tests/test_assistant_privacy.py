from tennis_ai.simulate import make_demo_sessions
from tennis_ai.assistant import build_evidence_packet

def test_evidence_packet_does_not_include_full_session_rows():
    d=make_demo_sessions(50)
    packet=build_evidence_packet(d)
    assert "sessions" not in packet
    assert "opponent" not in packet
