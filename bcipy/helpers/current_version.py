import pkg_resources
import os
import logging

from pathlib import Path
log = logging.getLogger(__name__)

def git_dir() -> Path:
    """Returns the root directory with the .git folder. If this source code 
    was not checked out from scm, answers None."""

    # Relative to current file; may need to be modified if method is moved.
    git_root = Path(os.path.abspath(__file__)).parent.parent.parent    
    git_meta = Path(os.path.join(git_root, '.git'))
    return os.path.abspath(git_meta) if git_meta.is_dir() else None

def git_hash() -> str:
    """Returns an abbreviated git sha hash if this code is being run from a
    git cloned version of bcipy; otherwise returns an empty string.
    
    Could also consider making a system call to:
    git describe --tags
    """

    git_path = git_dir()
    if not git_path:
        log.debug(".git path not found")
        return ""

    try:
        head_path = Path(os.path.join(git_path, "HEAD"))
        with open(head_path) as head_file:
            # First line contains a reference to the current branch.
            # ex. ref: refs/heads/branch_name
            ref = head_file.readline()

        ref_val = ref.split(":")[-1].strip()
        ref_path = Path(os.path.join(git_path, ref_val))
        with open(ref_path) as ref_file:
            sha = ref_file.readline()

        # sha hash is 40 characters; use an abbreviated 7-char version which
        # is displayed in github.
        return sha[0:7]
    except Exception as e:
        log.info("Error reading git version")
        log.info(e)
        return ""
        

def bcipy_version() -> str:
    """Gets the current bcipy version. If the current instance of bcipy is a
    git repository, appends the current abbreviated sha hash.
    """
    version = pkg_resources.get_distribution("bcipy").version
    sha_hash = git_hash()

    return f"{version} - {sha_hash}" if sha_hash else version

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='(%(threadName)-9s) %(message)s')
    print(bcipy_version())