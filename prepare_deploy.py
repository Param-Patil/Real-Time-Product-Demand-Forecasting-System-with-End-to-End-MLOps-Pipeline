import shutil
import os

def copy_models():
    source = "models"
    targets = ["api/models", "dashboards/models"]

    for target in targets:
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(source, target)
        print(f" Copied models to {target}")

if __name__ == "__main__":
    copy_models()
