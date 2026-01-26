import argparse
import os
import posixpath
import paramiko

from dotenv import load_dotenv
load_dotenv()

HOST = os.getenv("VM_HOST")
USER = os.getenv("VM_USER")
PWD  = os.getenv("VM_PASSWORD")
PORT = int(os.getenv("VM_PORT", "22"))

if not all([HOST, USER, PWD]):
    raise RuntimeError("Missing VM_HOST / VM_USER / VM_PASSWORD environment variables")

def ssh_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # force IPv4 if v6 is flaky
    c.connect(HOST, port=PORT, username=USER, password=PWD,
              timeout=20, auth_timeout=30, banner_timeout=45,
              look_for_keys=False, allow_agent=False)
    c.get_transport().set_keepalive(20)
    return c

def sftp_mkdir_p(sftp, remote_path: str):
    """Recursively mkdir like `mkdir -p` for POSIX paths, handling absolute paths correctly."""
    remote_path = posixpath.normpath(remote_path)
    if remote_path == "/":
        return

    # Split the path into components
    parts = remote_path.split("/")
    # If path is absolute, start from '/'
    if remote_path.startswith("/"):
        path = "/"
        start_index = 1
    else:
        path = ""
        start_index = 0

    for part in parts[start_index:]:
        if not part:
            continue
        if path in ("", "/"):
            path = posixpath.join(path, part) if path != "/" else "/" + part
        else:
            path = posixpath.join(path, part)
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def remote_exists_file(sftp, remote_path: str) -> bool:
    try:
        st = sftp.stat(remote_path)
        # treat as file if not a directory bit
        return not bool(st.st_mode & 0o040000)
    except FileNotFoundError:
        return False


def remote_exists_dir(sftp, remote_path: str) -> bool:
    try:
        st = sftp.stat(remote_path)
        return bool(st.st_mode & 0o040000)
    except FileNotFoundError:
        return False


def sftp_put_file(sftp, local_file: str, remote_dir: str):
    sftp_mkdir_p(sftp, remote_dir)
    remote_file = posixpath.join(remote_dir, os.path.basename(local_file))
    if remote_exists_file(sftp, remote_file):
        print(f"Skip: {remote_file}")
        return
    sftp.put(local_file, remote_file)
    print(f"Copied: {local_file} -> {remote_file}")


def sftp_put_dir(sftp, local_dir: str, remote_parent: str):
    """Upload a directory recursively, copying only files that don't yet exist remotely."""
    local_dir = os.path.abspath(local_dir)
    base = os.path.basename(local_dir.rstrip(os.sep))
    remote_dir = posixpath.join(remote_parent, base)

    # Ensure top-level target dir exists (do NOT skip if it already exists)
    sftp_mkdir_p(sftp, remote_dir)

    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        rel = "" if rel == "." else rel

        # Build remote directory path for this level
        if rel == "":
            rd = remote_dir
        else:
            # Ensure POSIX separators on remote
            rd = posixpath.join(remote_dir, rel.replace("\\", "/"))

        sftp_mkdir_p(sftp, rd)

        for fn in files:
            lf = os.path.join(root, fn)
            rf = posixpath.join(rd, fn)

            if remote_exists_file(sftp, rf):
                print(f"Skip: {rf}")
                continue

            sftp.put(lf, rf)
            print(f"Copied: {lf} -> {rf}")

def main():
    parser = argparse.ArgumentParser(description="Copy project files to remote machine")
    parser.add_argument("company", help="Company name (e.g., disney)")
    args = parser.parse_args()

    company = args.company
    projects_dir = company
    remote_root = f"/home/ssimon/github/config-space/data/{company}/projects"

    ssh = ssh_client()
    sftp = ssh.open_sftp()
    try:
        # Ensure base target exists
        sftp_mkdir_p(sftp, remote_root)

        for folder in os.listdir(projects_dir):
            folder_path = os.path.join(projects_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            # Skip if remote folder already exists
            remote_folder = posixpath.join(remote_root, folder)
            if remote_exists_dir(sftp, remote_folder):
                print(f"Skip: {remote_folder} (already exists)")
                continue

            sftp_mkdir_p(sftp, remote_folder)

            # Copy all JSON files from this folder
            for f in os.listdir(folder_path):
                if not f.endswith(".json"):
                    continue

                local_file = os.path.join(folder_path, f)
                sftp_put_file(sftp, local_file, remote_folder)

    finally:
        try:
            sftp.close()
        except Exception:
            pass
        ssh.close()

if __name__ == "__main__":
    main()