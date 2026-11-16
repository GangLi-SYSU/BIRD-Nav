#!/usr/bin/env python3
"""Show download coverage without counting intermediate copies twice."""
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent


def coverage(target, size, source, chunk):
    if target.exists() and target.stat().st_size == size:
        return size
    key = hashlib.sha256(str(target).encode()).hexdigest()
    folder = HERE/'.parts'/source/(key+'-'+str(chunk))
    total = 0
    for number in range((size+chunk-1)//chunk):
        lengths = [path.stat().st_size for path in (folder/str(number), folder/(str(number)+'.partial')) if path.exists()]
        tail = folder/(str(number)+'.tail')
        if tail.exists():
            partial = folder/(str(number)+'.partial')
            lengths.append(tail.stat().st_size+(partial.stat().st_size if partial.exists() else 0))
        total += min(max(lengths, default=0), min(chunk, size-number*chunk))
    return total


manifest = json.loads((HERE/'registry-manifest.json').read_text())
entries = [manifest['config'], *manifest['layers']]
image_size = sum(entry['size'] for entry in entries)
image_received = sum(coverage(HERE/'isaac-sim-4.2.0.oci'/'blobs'/'sha256'/entry['digest'].split(':')[1], entry['size'], 'registry', 32*1024*1024) for entry in entries)
if (HERE/'image-download.json').exists():
    image_received = image_size
print(f'Image: {image_received/2**30:.3f} / {image_size/2**30:.3f} GiB ({image_received/image_size:.1%})')
listing = HERE/'assets-file-list.json'
if listing.exists():
    entries = [entry for entry in json.loads(listing.read_text()) if not entry['directory']]
    asset_size = sum(entry['size'] for entry in entries)
    asset_received = 0
    for entry in entries:
        relative = unquote(entry['href'].split('/qardanQkDLwe7m2/', 1)[1])
        asset_received += coverage(HERE.parent/'isaac-sim_assets'/relative, entry['size'], 'assets', 8*1024*1024)
    print(f'Assets: {asset_received/2**30:.3f} / {asset_size/2**30:.3f} GiB ({asset_received/asset_size:.1%}); {len(entries)} files')
print('Complete:', (HERE/'download-complete.json').exists())
