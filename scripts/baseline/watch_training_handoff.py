"""Watch only an explicitly identified job; hand its pane to ER+GRPO on exit.

This does not terminate a live training job. A STOP file disables handoff.
Every event is flushed to the supplied local log by the caller.
"""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if stat[0] == 'Z' else stat[19]
    except FileNotFoundError:
        return None


def foreground(shell_pid):
    return int(Path(f'/proc/{shell_pid}/stat').read_text().split(') ', 1)[1].split()[5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--pane', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stop-file', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    stop = Path(args.stop_file)
    shell = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', args.pane, '#{pane_pid}']))
    pid = args.pid
    generation = 0
    while not stop.exists():
        birth = identity(pid)
        if birth is None:
            raise RuntimeError('Expected training process disappeared before identification')
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
        if not any(x in cmd for x in [b'train_pure_opd_sod.sh', b'continue_er_opd.sh']):
            raise RuntimeError('Refusing to watch an unrelated process')
        group = os.getpgid(pid)
        assert group == pid and group != os.getpgid(shell)
        started = time.monotonic()
        print(f'Watching GPU {args.gpu}, pane {args.pane}, job {pid}', flush=True)
        while identity(pid) == birth and not stop.exists():
            time.sleep(2)
        if stop.exists():
            break
        # This group was created by the validated launch, not a shared Ray server.
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(15):
            if foreground(shell) == os.getpgid(shell):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Pane is not idle; refusing to inject a command')
        # Do not create an unbounded restart storm if the fallback itself fails.
        if generation and time.monotonic() - started < 180:
            raise RuntimeError('ER fallback exited during startup; inspect its error before restarting')
        generation += 1
        name = f'er-fallback-gpu{args.gpu}-{args.tag}-{generation}'
        command = f'cd {shlex.quote(str(root))} && bash scripts/baseline/continue_er_opd.sh {args.gpu} {shlex.quote(name)}'
        subprocess.run(['tmux', 'send-keys', '-t', args.pane, '-l', command], check=True)
        subprocess.run(['tmux', 'send-keys', '-t', args.pane, 'Enter'], check=True)
        print(f'Handoff to {name}', flush=True)
        for _ in range(20):
            time.sleep(.2)
            pid = foreground(shell)
            if pid != os.getpgid(shell) and identity(pid) is not None:
                break
        else:
            raise RuntimeError('Fallback did not start')
    print('Handoff disabled by STOP file; live training left untouched', flush=True)


if __name__ == '__main__':
    main()
