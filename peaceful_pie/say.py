import argparse
from dataclasses import dataclass
from peaceful_pie.unity_comms import UnityComms

@dataclassd
class MyVector3:
    x: float
    y: float
    z: float

def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port=args.port, hostname="127.0.0.1")
    # unity_comms.Say(message=args.message)
    unity_comms.Move(translate=MyVector3(x=args.x, y=args.y, z=args.z))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--message", type=str, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=0)
    parser.add_argument("--z", type=float, default=0)
    args = parser.parse_args()
    run(args)

