import os
import posixpath
import paramiko

from dotenv import load_dotenv
load_dotenv()


PROJECTS_DIR = "projects"

HOST = os.getenv("VM_HOST")
USER = os.getenv("VM_USER")
PWD  = os.getenv("VM_PASSWORD")
PORT = int(os.getenv("VM_PORT", "22"))
REMOTE_ROOT = "/home/ssimon/github/config-space/data/projects"

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
    not_finished_projects = []

    ssh = ssh_client()
    sftp = ssh.open_sftp()
    try:
        # Ensure base target exists
        sftp_mkdir_p(sftp, REMOTE_ROOT)

        for folder in os.listdir(PROJECTS_DIR):
            folder_path = os.path.join(PROJECTS_DIR, folder)
            if not os.path.isdir(folder_path):
                continue

            files = os.listdir(folder_path)

            err_file = [f for f in files if f.endswith(".err")][0]

            with open(os.path.join(folder_path, err_file), "r") as f:
                content = f.read()
                if "Processing: 100%" not in content:
                    print(f"Skipping folder {folder_path} as job not finished.")
                    not_finished_projects.append(folder_path)
                    continue

            # Condition from your code:
            # If NO summary.json exists AND there IS any 'batch' file -> upload whole folder
            has_summary = any(f.endswith("summary.json") for f in files)
            has_batch   = any("batch" in f for f in files)
            has_last_commit = any(f.endswith("last_commit.json") for f in files)

            if (not has_summary) and has_batch:
                print(f"Copy entire folder {folder_path} to {REMOTE_ROOT}")
                sftp_put_dir(sftp, folder_path, REMOTE_ROOT)
                continue

            #if has_last_commit:
            #    project_name = folder
            #    last_commit_file = os.path.join(folder_path, f"{project_name}_last_commit.json")
            #    remote_dir = posixpath.join("/home/ssimon/github/config-space/data/projects_last_commit")
            #    print(f"Kopiere Datei {last_commit_file} nach {remote_dir}")
            #    sftp_put_file(sftp, last_commit_file, remote_dir)
            #    continue

            # Else: upload individual .json files except those with batch/summary in the name
            for f in files:
                try: 
                    if ("batch" in f) or ("summary" in f) or ("latest_commit" in f) or (not f.endswith(".json")):
                        continue
                    
                    project_name = os.path.basename(f).removesuffix(".json")
                    file_path = os.path.join(folder_path, f)

                    remote_dir = posixpath.join(REMOTE_ROOT, project_name)
                    sftp_put_file(sftp, file_path, remote_dir)
                except Exception as e:
                    print(f"Error copying files: {file_path}")
                    continue

    finally:
        try:
            sftp.close()
        except Exception:
            pass
        ssh.close()

        print("Not finished projects:")
        for project in not_finished_projects:
            print(project)

if __name__ == "__main__":
    main()