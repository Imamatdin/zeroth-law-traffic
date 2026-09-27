"""python -m demo.deploy --space ACCOUNT/zeroth-law-traffic-demo"""
import argparse
import os
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', required=True)
    parser.add_argument('--origin', help='Exact Vercel origin, when available')
    args = parser.parse_args()
    token = os.getenv('HF_TOKEN')
    if not token:
        raise SystemExit('Set HF_TOKEN to a Hugging Face write token, then rerun this command.')
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    root = Path(__file__).resolve().parents[1]
    # Only this deployment tree is uploaded; no weights, caches, private notes or credentials.
    with tempfile.TemporaryDirectory(prefix='zlt-space-') as tmp:
        stage = Path(tmp)
        for name in ('src', 'configs', 'demo'):
            shutil.copytree(root / name, stage / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        for name in ('solution.py', 'evaluate.py'):
            shutil.copy2(root / name, stage / name)
        shutil.copy2(root / 'demo/Dockerfile', stage / 'Dockerfile')
        shutil.copy2(root / 'demo/SPACE_README.md', stage / 'README.md')
        api.create_repo(args.space, repo_type='space', space_sdk='docker', exist_ok=True)
        if args.origin:
            api.add_space_variable(args.space, key='DEMO_CORS_ORIGINS', value=args.origin)
        api.upload_folder(repo_id=args.space, repo_type='space', folder_path=stage,
                          commit_message='Deploy Zeroth Law CPU demo')
    print(f'https://huggingface.co/spaces/{args.space}')


if __name__ == '__main__':
    main()
