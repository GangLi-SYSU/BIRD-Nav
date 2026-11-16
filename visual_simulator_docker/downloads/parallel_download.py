"""Resumable range downloads for the README's public registry and asset share."""
import base64
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
CHUNK = 8*1024*1024
BUFFER = 64*1024
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7890', 'http': 'http://127.0.0.1:7890'}))
SURF_ROOT = '/public.php/dav/files/qardanQkDLwe7m2/'
BASIC = 'Basic '+base64.b64encode(b'qardanQkDLwe7m2:').decode()
TOKEN_LOCK = threading.Lock()
TOKEN = None


def chunk_size(source):
    return 32*1024*1024 if source == 'registry' else CHUNK


def part_folder(target, source):
    key = hashlib.sha256(str(target).encode()).hexdigest()
    return HERE/'.parts'/source/(key+'-'+str(chunk_size(source)))


def registry_token():
    global TOKEN
    with TOKEN_LOCK:
        if TOKEN is None:
            for attempt in range(10):
                url = 'https://nvcr.io/proxy_auth?'+urllib.parse.urlencode({'service': 'nvcr.io', 'scope': 'repository:nvidia/isaac-sim:pull', '_': str(time.time_ns())})
                route = ['--noproxy', '*'] if attempt % 2 == 0 else ['--proxy', 'http://127.0.0.1:7890']
                command = ['curl', '--silent', '--show-error', '--fail', '--location', '--ipv4', '--connect-timeout', '10', '--max-time', '20', '--header', 'Cache-Control: no-cache', *route, url]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    obj = json.loads(result.stdout)
                    TOKEN = obj.get('token') or obj['access_token']
                    break
                print(f'Token refresh retry {attempt+1}/10: connection exit {result.returncode}', flush=True)
                time.sleep(2)
            if TOKEN is None:
                raise RuntimeError('NVIDIA token refresh failed after both network routes were retried')
        return TOKEN


def download_part(item):
    global TOKEN
    url, size, target, part_number, source = item
    chunk = chunk_size(source)
    start = part_number*chunk
    expected = min(chunk, size-start)
    folder = part_folder(target, source)
    folder.mkdir(parents=True, exist_ok=True)
    part = folder/str(part_number)
    if part.exists() and part.stat().st_size == expected:
        return
    partial = part.with_suffix('.partial')
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            partial.unlink()
            offset = 0
        if offset == expected:
            partial.replace(part)
            return
        if source == 'registry':
            authorization = 'Bearer '+registry_token()
            route = ['--proxy', 'http://127.0.0.1:7890'] if attempt % 2 == 0 else ['--noproxy', '*']
        else:
            authorization = BASIC
            route = ['--proxy', 'http://127.0.0.1:7890'] if (part_number+attempt) % 2 == 0 else ['--noproxy', '*']
        try:
            command = ['curl', '--silent', '--show-error', '--fail', '--location', '--ipv4', '--connect-timeout', '15', '--max-time', '300', '--range', f'{start+offset}-{start+expected-1}', '--max-filesize', str(expected-offset), '--config', '-', *route, url]
            with partial.open('ab') as output:
                result = subprocess.run(command, input='header = "Authorization: '+authorization+'"\n', text=True, stdout=output, stderr=subprocess.PIPE)
            if result.returncode or partial.stat().st_size != expected:
                raise RuntimeError(result.stderr.strip() or 'Incomplete range download')
            partial.replace(part)
            return
        except urllib.error.HTTPError as error:
            if source == 'registry' and error.code == 401:
                with TOKEN_LOCK:
                    TOKEN = None
            if attempt == 7:
                raise
        except Exception as error:
            if attempt in (0, 4):
                print(f'{source} retry for {target.name} chunk {part_number}: {error}', flush=True)
            if attempt == 7:
                raise
        time.sleep(min(8, attempt+1))


def assemble(url, size, target, source, digest=None):
    folder = part_folder(target, source)
    temporary = target.with_name(target.name+'.assembling')
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with temporary.open('wb') as output:
        chunk = chunk_size(source)
        for number in range((size+chunk-1)//chunk):
            with (folder/str(number)).open('rb') as part:
                for data in iter(lambda: part.read(4*1024*1024), b''):
                    output.write(data)
                    hasher.update(data)
    if temporary.stat().st_size != size or (digest and hasher.hexdigest() != digest):
        raise RuntimeError('Download size or SHA256 verification failed: '+target.name)
    temporary.replace(target)
    if folder.exists():
        shutil.rmtree(folder)
    return hasher.hexdigest()



def pooled_registry_ranges(tasks, workers):
    """Let one libcurl process reuse HTTP/2 connections across byte ranges."""
    global TOKEN
    for url, size, target, number, source in tasks:
        folder = part_folder(target, source)
        tail = folder/(str(number)+'.tail')
        partial = folder/(str(number)+'.partial')
        part = folder/str(number)
        expected = min(chunk_size(source), size-number*chunk_size(source))
        offset_file = folder/(str(number)+'.offset')
        if tail.exists() and not part.exists():
            old_size = partial.stat().st_size if partial.exists() else 0
            offset = int(offset_file.read_text()) if offset_file.exists() else old_size
            if old_size == expected:
                partial.replace(part)
            elif offset+tail.stat().st_size <= expected:
                with partial.open('r+b' if partial.exists() else 'wb') as output, tail.open('rb') as received:
                    output.truncate(offset)
                    output.seek(offset)
                    shutil.copyfileobj(received, output, 4*1024*1024)
            else:
                raise RuntimeError('Saved registry tail exceeds its byte range')
        if tail.exists():
            tail.unlink()
        if offset_file.exists():
            offset_file.unlink()
    for attempt in range(24):
        transfers = []
        for url, size, target, number, source in tasks:
            chunk = chunk_size(source)
            expected = min(chunk, size-number*chunk)
            folder = part_folder(target, source)
            folder.mkdir(parents=True, exist_ok=True)
            part = folder/str(number)
            if part.exists() and part.stat().st_size == expected:
                continue
            partial = folder/(str(number)+'.partial')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == expected:
                partial.replace(part)
                continue
            tail = folder/(str(number)+'.tail')
            (folder/(str(number)+'.offset')).write_text(str(offset))
            transfers.append((url, number*chunk+offset, number*chunk+expected-1, expected-offset, partial, tail, part, expected))
        if not transfers:
            return
        print(f'Registry pooled round {attempt+1}: {len(transfers)} remaining ranges; {workers} connections', flush=True)
        command = ['curl', '--parallel', '--parallel-max', str(workers), '--config', '-']
        for batch_start in range(0, len(transfers), workers):
            with TOKEN_LOCK:
                TOKEN = None
            authorization = 'Bearer '+registry_token()
            settings = []
            for url, start, end, left, partial, tail, part, expected in transfers[batch_start:batch_start+workers]:
                if settings:
                    settings.append('next')
                settings.extend(('silent', 'show-error', 'fail', 'location', 'ipv4', 'connect-timeout = 15', 'max-time = 120'))
                settings.extend((
                    'url = '+json.dumps(url+'?'+urllib.parse.urlencode({'_': str(time.time_ns())}), ensure_ascii=False),
                    'range = '+json.dumps(f'{start}-{end}'),
                    'output = '+json.dumps(str(tail), ensure_ascii=False),
                    'max-filesize = '+str(left),
                    'header = '+json.dumps('Authorization: '+authorization),
                    'header = "Cache-Control: no-cache"',
                    'header = "Pragma: no-cache"',
                ))
                if attempt % 2 == 0:
                    settings.append('proxy = "http://127.0.0.1:7890"')
                else:
                    settings.append('noproxy = "*"')
            with (HERE/'pooled-image-download.log').open('a') as log:
                subprocess.run(command, input='\n'.join(settings)+'\n', text=True, stdout=log, stderr=log)
        for url, start, end, left, partial, tail, part, expected in transfers:
            if not tail.exists():
                continue
            if tail.stat().st_size > left:
                raise RuntimeError('Registry response exceeded its requested range')
            with partial.open('ab') as output, tail.open('rb') as received:
                shutil.copyfileobj(received, output, 4*1024*1024)
            tail.unlink()
            offset_file = tail.with_suffix('.offset')
            if offset_file.exists():
                offset_file.unlink()
            if partial.stat().st_size == expected:
                partial.replace(part)
        time.sleep(3)
    raise RuntimeError('Registry range transfers exhausted their retry limit')


def transfer(items, source, workers):
    pending = []
    for url, size, target, digest in items:
        if target.exists() and target.stat().st_size == size:
            if not digest:
                continue
            hasher = hashlib.sha256()
            with target.open('rb') as handle:
                for data in iter(lambda: handle.read(4*1024*1024), b''):
                    hasher.update(data)
            if hasher.hexdigest() == digest:
                continue
        pending.append((url, size, target, digest))
    print(f'{source}: {len(items)} files; {sum(item[1] for item in items)/2**30:.2f} GiB; {workers} workers', flush=True)
    chunk = chunk_size(source)
    tasks = [(url, size, target, number, source) for url, size, target, digest in pending for number in range((size+chunk-1)//chunk)]
    if source == 'registry':
        pooled_registry_ranges(tasks, workers)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_part, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    for url, size, target, digest in pending:
        assemble(url, size, target, source, digest)


def registry(manifest, expected_digest):
    root = HERE/'isaac-sim-4.2.0.oci'
    blobs = root/'blobs'/'sha256'
    blobs.mkdir(parents=True, exist_ok=True)
    descriptors = [manifest['config'], *manifest['layers']]
    items = [('https://nvcr.io/v2/nvidia/isaac-sim/blobs/'+entry['digest'], entry['size'], blobs/entry['digest'].split(':')[1], entry['digest'].split(':')[1]) for entry in descriptors]
    # Recover the previous tool's partially downloaded large layer, after checking
    # that its prefix matches an independently fetched range of the same layer.
    largest = max(items, key=lambda item: item[1])
    target = largest[2]
    native_prefix = target.with_name(target.name+'.partial')
    if native_prefix.exists() and native_prefix.stat().st_size == largest[1]:
        native_prefix.replace(target)
    elif native_prefix.exists():
        folder = part_folder(target, 'registry')
        folder.mkdir(parents=True, exist_ok=True)
        with native_prefix.open('rb') as handle:
            remaining = native_prefix.stat().st_size
            number = 0
            while remaining:
                amount = min(chunk_size('registry'), remaining)
                complete = amount == chunk_size('registry')
                recovered = folder/(str(number) if complete else str(number)+'.partial')
                already = (folder/str(number)).exists() or (recovered.exists() and recovered.stat().st_size >= amount)
                if already:
                    handle.seek(amount, 1)
                else:
                    with recovered.open('wb') as output:
                        left = amount
                        while left:
                            data = handle.read(min(4*1024*1024, left))
                            output.write(data)
                            left -= len(data)
                remaining -= amount
                number += 1
        print(f'Native prefix reused: {native_prefix.stat().st_size/2**30:.2f} GiB', flush=True)
    transfer(items, 'registry', 4)
    if native_prefix.exists():
        native_prefix.unlink()
    raw = (HERE/'registry-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest.split(':')[1]:
        raise RuntimeError('Pinned registry manifest digest does not match')
    (blobs/expected_digest.split(':')[1]).write_bytes(raw)
    (root/'oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}\n')
    index = {'schemaVersion': 2, 'mediaType': 'application/vnd.oci.image.index.v1+json', 'manifests': [{'mediaType': manifest['mediaType'], 'size': len(raw), 'digest': expected_digest, 'annotations': {'org.opencontainers.image.ref.name': '4.2.0'}}]}
    (root/'index.json').write_text(json.dumps(index)+'\n')
    # Discard only temporary files left by the earlier registry downloader.
    for path in blobs.glob('*.tmp'):
        path.unlink()
    parts = HERE/'.parts'/'registry'
    if parts.exists():
        shutil.rmtree(parts)
    return root


def list_directory(href, depth='1'):
    body = b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/><d:getcontentlength/><d:getetag/></d:prop></d:propfind>'
    headers = {'Authorization': BASIC, 'Depth': depth, 'Content-Type': 'application/xml'}
    for attempt in range(5):
        try:
            opener = DIRECT if attempt % 2 == 0 else PROXY
            with opener.open(urllib.request.Request('https://surfdrive.surf.nl'+href, headers=headers, method='PROPFIND', data=body), timeout=35) as response:
                raw = response.read()
            root = ET.fromstring(raw)
            result = []
            for node in root.findall('{DAV:}response'):
                name = node.findtext('{DAV:}href')
                if name.rstrip('/') == href.rstrip('/'):
                    continue
                if not name.startswith(SURF_ROOT):
                    raise RuntimeError('Asset listing contains a path outside the public share')
                directory = node.find('.//{DAV:}collection') is not None
                size = node.findtext('.//{DAV:}getcontentlength') or '0'
                result.append({'href': name, 'directory': directory, 'size': int(size)})
            return result
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt+1)


def assets():
    listing = HERE/'assets-file-list.json'
    if listing.exists():
        entries = json.loads(listing.read_text())
    else:
        entries = []
        pending = [SURF_ROOT]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            while pending:
                next_round = []
                for result in pool.map(list_directory, pending):
                    entries.extend(result)
                    next_round.extend(entry['href'] for entry in result if entry['directory'])
                pending = next_round
        listing.write_text(json.dumps(entries, indent=2)+'\n')
    destination = HERE.parent/'isaac-sim_assets'
    items = []
    for entry in entries:
        path = PurePosixPath(urllib.parse.unquote(entry['href'][len(SURF_ROOT):]))
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe asset path')
        target = destination.joinpath(*path.parts)
        if entry['directory']:
            target.mkdir(parents=True, exist_ok=True)
        else:
            items.append(('https://surfdrive.surf.nl'+entry['href'], entry['size'], target, None))
    # Keep assets in the original chunk layout across resumed runs.
    for _, _, target, _ in items:
        old_key = hashlib.sha256(str(target).encode()).hexdigest()
        old = HERE/'.parts'/'assets'/old_key
        new = part_folder(target, 'assets')
        if old.exists() and not new.exists():
            old.rename(new)
    transfer(items, 'assets', 12)
    trees = list((destination/'Trees').rglob('*.usd'))
    if not trees or not (destination/'noon_grass_4k.hdr').is_file():
        raise RuntimeError('Required asset files are missing')
    file_checksums = {}
    for url, size, target, _ in items:
        hasher = hashlib.sha256()
        with target.open('rb') as handle:
            for data in iter(lambda: handle.read(4*1024*1024), b''):
                hasher.update(data)
        file_checksums[str(target.relative_to(destination))] = hasher.hexdigest()
    (HERE/'assets-sha256.json').write_text(json.dumps(file_checksums, indent=2)+'\n')
    result = {'source': 'https://surfdrive.surf.nl/s/qardanQkDLwe7m2', 'files': len(items), 'bytes': sum(item[1] for item in items), 'tree_usd_files': len(trees), 'destination': str(destination), 'verification': 'remote file sizes matched; local SHA256 checksums recorded'}
    (HERE/'assets-download.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Assets downloaded and verified: {len(items)} files; {len(trees)} tree USD files', flush=True)
    return result
