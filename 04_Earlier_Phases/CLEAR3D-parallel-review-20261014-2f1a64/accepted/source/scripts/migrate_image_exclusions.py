"""No default execution. Independent review and separate authorization required."""
import argparse
from lidc3d.derivative import migrate_exclusion

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True)
    p.add_argument('--ledger',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--ledger-sha256',required=True,help='Exact independently reviewed ledger hash')
    p.add_argument('--execute',action='store_true',help='Use only after separate migration authorization')
    p.add_argument('--reuse-mode',choices=['copy','reflink-auto'],default='copy')
    args=p.parse_args()
    migrate_exclusion(args.parent,args.ledger,args.output,execute=args.execute,reuse_mode=args.reuse_mode,ledger_sha256=args.ledger_sha256)

if __name__=='__main__':
    main()
