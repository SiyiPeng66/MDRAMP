"""Single entry point for release workflows and advanced engineering commands."""
import sys

def main():
    argv=sys.argv[1:]
    if not argv or argv in (["--help"],["-h"]):
        from mdramp.release_cli import parser
        parser().print_help();return 0
    if argv[0] in {"train","calibrate","predict","generate"} and (len(argv)==1 or argv[1].startswith("-")):
        from mdramp.release_cli import main as release_main
        return release_main(argv)
    from mdramp.cli import main as advanced_main
    return advanced_main(argv)

if __name__=="__main__":raise SystemExit(main())
