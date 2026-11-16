#!/usr/bin/env python3
"""Download the README's assets and pinned Isaac Sim image to this directory."""

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import signal
import sys
import time
import urllib.request
from pathlib import Path
import parallel_download

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
IMAGE = 'nvcr.io/nvidia/isaac-sim:4.2.0'
DIGEST = 'sha256:b606646df3aab3f38ba655ce96cdf8f45a13bb167f01c4b2992e40d6f2900089'
TOOL_URL = 'https://github.com/regclient/regclient/releases/download/v0.11.6/regctl-linux-amd64'
TOOL_SHA = '8e0e62a497fcdb8048d18aa927a139613176ba0531f412bc541044e28f9856bd'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ENV = {key: value for key, value in os.environ.items() if key.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
BLOCK = 4*1024*1024


def total_file_bytes(paths):
    total = 0
    try:
        for path in paths:
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        pass
    return total


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            result.update(block)
    return result.hexdigest()


def fetch(url, destination, headers=None, use_proxy=False):
    partial = destination.with_name(destination.name+'.partial')
    opener = urllib.request.build_opener() if use_proxy else OPENER
    for attempt in range(1, 6):
        try:
            with opener.open(urllib.request.Request(url, headers=headers or {}), timeout=60) as response:
                with partial.open('wb') as handle:
                    shutil.copyfileobj(response, handle, BLOCK)
            partial.replace(destination)
            return
        except Exception as error:
            print(f'Download retry {attempt}/5 for {destination.name}: {type(error).__name__}: {error}', flush=True)
            if attempt == 5:
                raise
            time.sleep(5)


def install_tool():
    tool = HERE/'tools'/'regctl'
    tool.parent.mkdir(exist_ok=True)
    if not tool.exists() or sha256(tool) != TOOL_SHA:
        fetch(TOOL_URL, tool, use_proxy=True)
    if sha256(tool) != TOOL_SHA:
        raise RuntimeError('regctl checksum does not match its official release')
    tool.chmod(0o755)
    print('regctl v0.11.6: official SHA256 verified', flush=True)
    return tool


def download_assets():
    return parallel_download.assets()


def download_image():
    tool = install_tool()
    local = HERE/'isaac-sim-4.2.0.oci'
    archive = HERE/'isaac-sim-4.2.0.docker.tar'
    manifest_file = HERE/'image-download.json'
    if archive.exists() and manifest_file.exists():
        result = json.loads(manifest_file.read_text())
        if sha256(archive) == result['archive_sha256']:
            print('Previously downloaded image archive: SHA256 verified', flush=True)
            return result
    log_path = HERE/'image-download.log'
    manifest = json.loads((HERE/'registry-manifest.json').read_text())
    parallel_download.registry(manifest, DIGEST)
    print('Image layers downloaded; verifying OCI blob checksums', flush=True)
    count = 0
    for blob in (local/'blobs'/'sha256').iterdir():
        if sha256(blob) != blob.name:
            raise RuntimeError('Image blob checksum mismatch: '+blob.name)
        count += 1
    partial = archive.with_name(archive.name+'.partial')
    subprocess.run([str(tool), 'image', 'export', '--name', IMAGE, 'ocidir://'+str(local)+':4.2.0', str(partial)], env=ENV, check=True)
    partial.replace(archive)
    result = {'source': IMAGE, 'manifest_digest': DIGEST, 'platform': 'linux/amd64', 'verified_blobs': count, 'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'archive_sha256': sha256(archive), 'download_tool_sha256': TOOL_SHA}
    manifest_file.write_text(json.dumps(result, indent=2)+'\n')
    # These are only this script's intermediate blobs; the verified archive is retained.
    shutil.rmtree(local)
    print('Image exported: Docker-loadable archive and SHA256 manifest saved', flush=True)
    return result


def main():
    if '--restart' in sys.argv or '--stop' in sys.argv:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                args = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
                if len(args) < 3 or args[1:3] != ['-u', str(Path(__file__).resolve())]:
                    continue
                children = set()
                for task in (proc/'task').iterdir():
                    children.update(int(value) for value in (task/'children').read_text().split())
                os.kill(int(proc.name), signal.SIGTERM)
                for child in children:
                    try:
                        os.kill(child, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                print('Previous download process stopped; saved files will be reused', flush=True)
            except (OSError, UnicodeError):
                continue
        if '--stop' in sys.argv:
            print('Download processes stopped; all saved files are retained', flush=True)
            return
        # Stop only native transfers started by this downloader, including any
        # children orphaned by a previous interrupted run.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
                if not args or Path(args[0]).name != 'curl' or '--config' not in args:
                    continue
                if args[-1].startswith(('https://nvcr.io/v2/nvidia/isaac-sim/blobs/', 'https://surfdrive.surf.nl/public.php/dav/files/qardanQkDLwe7m2/')):
                    os.kill(int(proc.name), signal.SIGTERM)
            except (OSError, UnicodeError):
                continue
    if shutil.disk_usage(HERE).free < 24*2**30:
        raise RuntimeError('At least 24 GiB free space is needed for downloads and export')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {'assets': pool.submit(download_assets), 'image': pool.submit(download_image)}
        started = time.monotonic()
        while not all(task.done() for task in tasks.values()):
            done, _ = concurrent.futures.wait(tasks.values(), timeout=30)
            partials = list(HERE.glob('*.partial'))
            oci = HERE/'isaac-sim-4.2.0.oci'/'blobs'/'sha256'
            downloaded = total_file_bytes(partials)
            if oci.exists():
                downloaded += total_file_bytes(oci.iterdir())
            parts = HERE/'.parts'
            if parts.exists():
                downloaded += total_file_bytes(parts.rglob('*'))
            status = ', '.join(name+(' done' if task.done() else ' running') for name, task in tasks.items())
            print(f'Progress: {status}; active download files {downloaded/2**30:.2f} GiB; elapsed {(time.monotonic()-started)/60:.1f} min; disk free {shutil.disk_usage(HERE).free/2**30:.1f} GiB', flush=True)
        results = {name: task.result() for name, task in tasks.items()}
    (HERE/'download-complete.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Both README downloads completed and verified.', flush=True)


if __name__ == '__main__':
    main()
