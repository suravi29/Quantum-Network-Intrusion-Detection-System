# Attack categories for NSL-KDD

DOS_ATTACKS = [
    'back', 'land', 'neptune', 'pod',
    'smurf', 'teardrop'
]

PROBE_ATTACKS = [
    'ipsweep', 'nmap', 'portsweep', 'satan'
]

R2L_ATTACKS = [
    'ftp_write', 'guess_passwd',
    'imap', 'multihop',
    'phf', 'spy',
    'warezclient', 'warezmaster'
]

U2R_ATTACKS = [
    'buffer_overflow',
    'loadmodule',
    'perl',
    'rootkit'
]


def map_attack(label):
    if label == "normal":
        return "Normal"

    elif label in DOS_ATTACKS:
        return "DoS"

    elif label in PROBE_ATTACKS:
        return "Probe"

    elif label in R2L_ATTACKS:
        return "R2L"

    elif label in U2R_ATTACKS:
        return "U2R"

    else:
        return "Unknown"