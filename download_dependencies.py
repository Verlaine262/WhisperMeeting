"""Download dependencies from their publishers. Never uploads local files."""
import argparse,hashlib,json,shutil,tempfile,urllib.request,zipfile
from pathlib import Path

def download(url,path):
    with urllib.request.urlopen(url,timeout=300) as response,path.open('wb') as out:
        shutil.copyfileobj(response,out)

def main():
    parser=argparse.ArgumentParser(description='Download official Whisper model and NVIDIA DLLs')
    parser.add_argument('--destination',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.destination.resolve();root.mkdir(parents=True,exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='dropbox-dash/faster-whisper-large-v3-turbo',local_dir=root/'model-turbo',allow_patterns=['model.bin','config.json','tokenizer.json','vocabulary.json','preprocessor_config.json','LICENSE*','README.md'])
    cuda=root/'cuda';cuda.mkdir(exist_ok=True)
    for name,version in [('nvidia-cublas-cu12','12.9.2.10'),('nvidia-cudnn-cu12','9.9.0.52')]:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as response:
            metadata=json.load(response)
        asset=next(f for f in metadata['urls'] if f['filename'].endswith('win_amd64.whl'))
        with tempfile.TemporaryDirectory(prefix='dependency-',dir=root) as temp:
            wheel=Path(temp)/asset['filename'];print('Downloading',asset['filename'],flush=True);download(asset['url'],wheel)
            with wheel.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=asset['digests']['sha256']:raise RuntimeError('SHA256 mismatch: '+asset['filename'])
            with zipfile.ZipFile(wheel) as archive:
                for item in archive.namelist():
                    if item.lower().endswith('.dll'):(cuda/Path(item).name).write_bytes(archive.read(item))
                    elif not item.endswith('/') and any(k in item.lower() for k in ['license','eula','notice']):
                        target=root/'third-party-licenses'/name/Path(item).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(item))
    print('Dependencies ready in',root)

if __name__=='__main__':main()
