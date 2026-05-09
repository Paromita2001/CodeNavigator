import git
import os

def clone_repo(url: str, dest: str) -> tuple[bool, str]:
    """
    Clones a GitHub repository to the destination path.
    Returns (True, "") on success or (False, error_message) on failure.
    """
    try:
        os.makedirs(dest, exist_ok=True)
        git.Repo.clone_from(url, dest)
        return True, ""
    except git.exc.GitCommandError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)